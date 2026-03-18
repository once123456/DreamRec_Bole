# 核心导入：仅使用 RecBole 原生工具
import sys
import json
from logging import getLogger
import torch

# RecBole 核心工具（原生）
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_environment,
    get_local_time
)
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer.trainer import TraditionalTrainer

# 导入你的 DreamRec 模型
from dreamrec import DreamRec

if __name__ == '__main__':
    # 1. 配置初始化
    config = Config(
        model=DreamRec,
        config_file_list=['config.yaml'],
    )

    # 2. 随机种子初始化
    init_seed(config['seed'], config['reproducibility'])
    
    # 3. 日志初始化
    init_logger(config)
    logger = getLogger()
    
    # 4. 打印运行环境信息
    env_info = get_environment(config)
    logger.info(set_color("运行环境信息", "green") + f": \n{env_info}")

    # 5. 数据集加载（原生RecBole）
    logger.info(set_color("开始加载数据集...", "blue"))
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info(
        set_color("数据集加载完成", "green") +
        f" | 训练集批次: {len(train_data)} | 验证集批次: {len(valid_data)} | 测试集批次: {len(test_data)}" +
        f" | 物品总数: {dataset.item_num} | 用户总数: {dataset.user_num}"
    )

    # 6. 模型初始化
    logger.info(set_color("开始初始化模型...", "blue"))
    model = DreamRec(config, dataset).to(config['device'])
    logger.info(set_color("模型结构", "green") + f": \n{model}")
    
    # 7. 跳过FLOPs计算（低版本RecBole兼容）
    try:
        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(set_color("可训练参数量", "green") + f": {param_num:,}")
    except Exception as e:
        logger.warning(set_color(f"计算参数量失败: {str(e)}", "yellow"))

    # 8. 训练器初始化
    trainer = TraditionalTrainer(config, model)

    # 9. 模型训练（移除early_stop参数，用配置文件控制早停）
    logger.info(set_color("开始模型训练...", "blue"))
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        show_progress=config["show_progress"]  # 仅保留show_progress参数
    )

    # 10. 模型测试
    logger.info(set_color("开始模型测试...", "blue"))
    test_result = trainer.evaluate(
        test_data,
        show_progress=config["show_progress"],
        load_best_model=True
    )

    # 11. 结果输出
    logger.info(set_color("===== 最终结果 =====", "red"))
    logger.info(set_color("最佳验证结果", "yellow") + f": {best_valid_result}")
    logger.info(set_color("测试集结果", "yellow") + f": {test_result}")

    # 12. 保存结果（适配Config类）
    if 'save_result' in config.final_config_dict and config['save_result']:
        result_path = f'result_{get_local_time()}.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': config.final_config_dict,
                'best_valid': best_valid_result,
                'test': test_result,
                'param_num': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }, f, ensure_ascii=False, indent=4)
        logger.info(set_color(f"测试结果已保存到 {result_path}", "green"))

    # 13. 保存模型
    if 'save_model' in config.final_config_dict and config['save_model']:
        save_path = trainer.save_model()
        logger.info(set_color(f"最优模型已保存到 {save_path}", "green"))