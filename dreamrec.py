import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
#from recbole.model.sequential_recommender import get_attention_mask  # 补充：RecBole 掩码工具（因果掩码核心）
import numpy as np
import math


class DreamRec(SequentialRecommender):
    """
    DreamRec: Guided Diffusion for Sequential Recommendation
    优化：区分训练/推理阶段，仅训练时启用随机mask
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

        # ── 第一步：指导向量参数 ─────────────────────────
        self.p = config['p']  # classifier-free guidance drop 概率

        # ── Embedding 层 ─────────────────────────────────
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0) #原版使用item_num作为padding——idx
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        nn.init.normal_(self.item_embedding.weight, 0, self.initializer_range)
        # ── none_embedding（原版用于 drop 时的无条件向量）
        self.none_embedding = nn.Embedding(1, self.hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, self.initializer_range)

        # ── Transformer 编码器（对齐原版 mh_attn + feed_forward + ln_1/2/3）
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            
        )
        # ── 原版 LN/Dropout 组件（对齐顺序） ─────────────────────
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)  # 原版 emb_dropout
        #不需要self.ln_1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)  transfomer当中已经实现了

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
        Args:
            item_seq: 用户行为序列 [B, L]
            item_seq_len: 序列真实长度 [B]
            enable_drop: 手动控制是否启用drop（优先级高于self.training）
        Returns:
            h: 指导向量 [B, H]
        """
        B, L = item_seq.size()
        device = item_seq.device  # 优化1：提前定义device，简化后续代码
        print(f"[Forward] 输入序列形状: item_seq={item_seq.shape}, item_seq_len={item_seq_len[:5]}（前5个长度）")
        
        # 1. item + pos embedding（对齐原版，优化位置编码生成逻辑）
        # 优化2：简化position_ids生成，减少内存拷贝
        position_ids = torch.arange(L, dtype=torch.long, device=device).repeat(B, 1)
        item_emb = self.item_embedding(item_seq)        # [B,L,H]
        pos_emb = self.position_embedding(position_ids) # [B,L,H]
        input_emb = item_emb + pos_emb                  # 原版直接加

        print(f"[Forward] Embedding形状: item_emb={item_emb.shape}, pos_emb={pos_emb.shape}")
        # 2. 原版 emb_dropout（先 Dropout，再 Mask，对齐原版顺序）
        seq = self.emb_dropout(input_emb)

        # 3. 原版 Padding Mask（RecBole 适配：pad=0 而非 item_num）
        # 原因：RecBole 数据集默认 pad=0，适配框架更高效
        mask = torch.ne(item_seq, 0).float().unsqueeze(-1).to(device)  # 优化3：复用提前定义的device
        seq *= mask

        # 打印3：Mask 有效性（统计有效位置占比）
        valid_ratio = mask.sum() / mask.numel()
        print(f"[Forward] Padding Mask: 有效位置占比={valid_ratio:.4f}")

        # 4. 关键补充：生成 RecBole 因果+Padding 注意力掩码（对齐原版因果mask）
        # bidirectional=False → 下三角掩码，禁止关注未来位置（原版核心逻辑）
        # 注：原版是Q=归一化seq，K/V=原始seq；RecBole Transformer不支持拆分Q/K/V，统一用seq作为输入
        attn_mask = self.get_attention_mask(item_seq, bidirectional=False)
        
        # 5. Transformer（对齐原版 mh_attn + feed_forward + ln_2/3）
        # 优化4：移除冗余的ln_1，避免重复归一化（RecBole Transformer内部已做Pre-LN）
        trm_out = self.trm_encoder(seq, attn_mask, output_all_encoded_layers=True)
        output = trm_out[-1]                            # [B,L,H]

        # 打印4：Transformer 输出信息
        print(f"[Forward] Transformer输出形状: {output.shape}, 输出均值={output.mean():.4f}, 方差={output.var():.4f}")

        # 6. 再次Padding Mask过滤（对齐原版 ff_out *= mask）
        # 即使Transformer内部处理了mask，再次乘mask确保padding位置完全置零（原版核心逻辑）
        output *= mask   

        # 7. 取最后一个位置（对齐原版 extract_axis_1(ff_out, len_states-1)）
        # 优化5：鲁棒性增强，处理极端情况（如item_seq_len=0）
        valid_indices = torch.clamp(item_seq_len - 1, min=0)  # 确保索引≥0
        h = self.gather_indexes(output, valid_indices)        # [B,H]
        # 打印5：最终指导向量信息
        print(f"[Forward] 指导向量形状: {h.shape}, 非零元素占比={(h!=0).sum()/h.numel():.4f}")
        
        # 8. 核心优化：仅训练阶段启用 classifier-free guidance drop
        # 优先级：手动传参 enable_drop > 模型自身 training 状态
        if (enable_drop is True) or (enable_drop is None and self.training):
            # 原版mask生成公式：(torch.sign(rand - p) + 1)/2，替代简单的 < p
            mask1d = (torch.sign(torch.rand(B, device=device) - self.p) + 1) / 2
            drop_mask = mask1d.view(B, 1).expand(B, self.hidden_size)  # [B,H]
            none_emb_batch = self.none_embedding.weight[0:1].expand(B, -1)  # [B,H]
            h = h * drop_mask + none_emb_batch * (1 - drop_mask)  # drop 时用 none_emb
            # 打印6：Dropout 信息
            print(f"[Forward] 训练阶段dropout: drop_mask有效占比={drop_mask.mean():.4f}")
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