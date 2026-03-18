import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import LightTransformerEncoder  # 替换为LightTransformerEncoder
from recbole.model.loss import get_attention_mask
import numpy as np
import math


class DreamRec(SequentialRecommender):
    """
    DreamRec: Guided Diffusion for Sequential Recommendation
    优化：区分训练/推理阶段，仅训练时启用随机mask
    核心升级：使用RecBole专为序列推荐优化的LightTransformerEncoder
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ── 基础参数 ─────────────────────────────────────
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']

        # ── LightTransformerEncoder 新增核心参数 ─────────
        self.k_interests = config.get('k_interests', 5)  # 低秩分解兴趣数（RecBole默认5）
        self.seq_len = self.max_seq_length  # 序列最大长度（对应LightTransformer的seq_len）

        # ── 第一步：指导向量参数 ─────────────────────────
        self.p = config['p']  # classifier-free guidance drop 概率

        # ── Embedding 层 ─────────────────────────────────
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0) #原版使用item_num作为padding——idx
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  # LightTransformer需要单独的位置Embedding
        
        nn.init.normal_(self.item_embedding.weight, 0, self.initializer_range)
        # ── none_embedding（原版用于 drop 时的无条件向量）
        self.none_embedding = nn.Embedding(1, self.hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, self.initializer_range)

        # ── LightTransformer 编码器（序列推荐专用，对齐原版注意力逻辑）
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,  # 新增：低秩分解兴趣数
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,          # 新增：序列最大长度
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        # ── 仅保留原版 emb_dropout（LightTransformer内部已处理LN）
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)  # 原版 emb_dropout

        # ── 原有的占位（后面用） ─────────────────────────
        self.proj = nn.Linear(self.hidden_size, self.n_items)
        self.loss_fct = nn.CrossEntropyLoss()

        # ── Diffusion 占位（第一步不改） ─────────────────
        self.timesteps = config['timesteps']
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.beta_sche = config['beta_sche']
        self.w = config['w']
        self.diffuser_type = config['diffuser_type']

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            

    def forward(self, item_seq, item_seq_len, enable_drop: bool = None):
        """
        优化版 forward：精准区分训练/推理阶段的随机mask
        适配LightTransformerEncoder：位置编码单独传入，而非加和到Embedding
        Args:
            item_seq: 用户行为序列 [B, L]
            item_seq_len: 序列真实长度 [B]
            enable_drop: 手动控制是否启用drop（优先级高于self.training）
        Returns:
            h: 指导向量 [B, H]
        """
        B, L = item_seq.size()

        # 1. item embedding + position embedding（LightTransformer要求单独生成pos_emb）
        position_ids = torch.arange(L, dtype=torch.long, device=item_seq.device).unsqueeze(0).expand(B, -1)
        item_emb = self.item_embedding(item_seq)        # [B,L,H]
        pos_emb = self.position_embedding(position_ids) # [B,L,H]（单独传入LightTransformer，不加和）

        # 2. 原版 emb_dropout（仅作用于item_emb，pos_emb由LightTransformer内部处理）
        seq = self.emb_dropout(item_emb)

        # 3. 原版 Padding Mask（RecBole 适配：pad=0 而非 item_num）
        mask = torch.ne(item_seq, 0).float().unsqueeze(-1).to(item_seq.device)
        seq *= mask

        # 4. LightTransformer编码（核心：pos_emb单独传入，内部处理解耦位置编码）
        # LightTransformer天然适配单向序列，无需额外attn_mask实现因果约束
        trm_out = self.trm_encoder(seq, pos_emb, output_all_encoded_layers=True)
        output = trm_out[-1]                            # [B,L,H]

        # 5. 再次Padding Mask过滤（对齐原版 ff_out *= mask）
        output *= mask

        # 6. 取最后一个位置（对齐原版 extract_axis_1(ff_out, len_states-1)）
        h = self.gather_indexes(output, item_seq_len - 1)  # [B,H]

        # 7. 核心优化：仅训练阶段启用 classifier-free guidance drop
        # 优先级：手动传参 enable_drop > 模型自身 training 状态
        if (enable_drop is True) or (enable_drop is None and self.training):
            # 原版mask生成逻辑
            mask1d = (torch.sign(torch.rand(B, device=h.device) - self.p) + 1) / 2
            drop_mask = mask1d.view(B, 1).expand(B, self.hidden_size)  # [B,H]
            none_emb_batch = self.none_embedding.weight[0:1].expand(B, -1)  # [B,H]
            h = h * drop_mask + none_emb_batch * (1 - drop_mask)  # drop 时用 none_emb
        # 推理阶段：直接返回纯净的h，不做任何mask

        return h

    def calculate_loss(self, interaction):
        """训练阶段：自动启用drop"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        # 训练阶段：enable_drop=True（或不传，依赖self.training）
        seq_output = self.forward(item_seq, item_seq_len)  # [B, H]，带drop的h
        logits = self.proj(seq_output)
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        """推理阶段：强制关闭drop"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # 推理阶段：手动传 enable_drop=False，确保关闭随机mask
        seq_output = self.forward(item_seq, item_seq_len, enable_drop=False)  # [B, H]，纯净的h
        
        all_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, all_item_emb.t())
        return scores