# 核心导入：仅使用 RecBole 原生工具 
import sys
import json
import os
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
    # ========== 全局异常捕获：避免程序崩溃 ==========
    try:
        # 1. 配置初始化（RecBole 原生）
        config = Config(
            model=DreamRec,
            config_file_list=['config.yaml'],
        )

        # 2. 随机种子初始化（保证复现性）
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
        
        # 7. 计算可训练参数量（缓存结果，避免重复计算）
        param_num = 0
        try:
            param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(set_color("可训练参数量", "green") + f": {param_num:,}")
        except Exception as e:
            logger.warning(set_color(f"计算参数量失败: {str(e)}", "yellow"))
            param_num = "unknown"

        # 8. 训练器初始化（RecBole 原生 TraditionalTrainer 适配 DreamRec）
        logger.info(set_color("初始化训练器...", "blue"))
        trainer = TraditionalTrainer(config, model)

        # 9. 模型训练（用配置文件控制早停，RecBole 原生逻辑）
        logger.info(set_color("开始模型训练...", "blue"))
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            show_progress=config["show_progress"]
        )
        logger.info(set_color("模型训练完成", "green"))

        # 10. 模型测试（加载最优模型）
        logger.info(set_color("开始模型测试...", "blue"))
        test_result = trainer.evaluate(
            test_data,
            show_progress=config["show_progress"],
            load_best_model=True  # 加载验证集最优模型
        )
        logger.info(set_color("模型测试完成", "green"))

        # 11. 结果输出（格式化展示）
        logger.info(set_color("===== 最终结果 =====", "red"))
        logger.info(set_color("最佳验证结果", "yellow") + f": \n{json.dumps(best_valid_result, indent=4)}")
        logger.info(set_color("测试集结果", "yellow") + f": \n{json.dumps(test_result, indent=4)}")

        # 12. 保存结果（适配 RecBole Config 规范）
        if config.get('save_result', False):
            # 创建结果目录（避免路径不存在）
            os.makedirs('results', exist_ok=True)
            result_path = f'results/dreamrec_{config["dataset"]}_{get_local_time()}.json'
            # 构建结果字典（避免设备不匹配）
            result_dict = {
                'model': 'DreamRec',
                'dataset': config['dataset'],
                'config': config.final_config_dict,  # 完整配置
                'best_valid': best_valid_result,
                'test': test_result,
                'param_num': param_num,
                'train_batch_num': len(train_data),
                'valid_batch_num': len(valid_data),
                'test_batch_num': len(test_data)
            }
            # 保存结果
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            logger.info(set_color(f"测试结果已保存到 {result_path}", "green"))

        # 13. 保存模型（RecBole 原生逻辑）
        if config.get('save_model', False):
            save_path = trainer.save_model()
            logger.info(set_color(f"最优模型已保存到 {save_path}", "green"))

    except Exception as e:
        # 捕获所有异常，记录日志并退出
        logger.error(set_color(f"程序运行出错: {str(e)}", "red"))
        # 打印异常栈（便于调试）
        import traceback
        logger.error(set_color(f"异常栈: \n{traceback.format_exc()}", "red"))
        sys.exit(1)

    # 程序正常结束
    logger.info(set_color("===== 程序正常结束 =====", "green"))
    sys.exit(0)