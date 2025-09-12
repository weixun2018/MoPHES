import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def extract_training_config(content):
    """从日志中提取训练配置"""
    config = {}
    
    # 提取命令行参数
    cmd_pattern = r"cmd = .* finetune\.py (.+)"
    cmd_match = re.search(cmd_pattern, content)
    
    if cmd_match:
        cmd_args = cmd_match.group(1)
        
        # 提取关键参数
        params = {
            'learning_rate': r'--learning_rate (\S+)',
            'max_steps': r'--max_steps (\d+)',
            'warmup_steps': r'--warmup_steps (\d+)',
            'eval_steps': r'--eval_steps (\d+)',
            'logging_steps': r'--logging_steps (\d+)',
            'per_device_train_batch_size': r'--per_device_train_batch_size (\d+)',
            'gradient_accumulation_steps': r'--gradient_accumulation_steps (\d+)',
        }
        
        for param, pattern in params.items():
            match = re.search(pattern, cmd_args)
            if match:
                value = match.group(1)
                config[param] = float(value) if '.' in value or 'e' in value.lower() else int(value)
    
    return config

def extract_loss_from_log(log_file_path):
    """从训练日志中提取loss数据"""
    train_losses = []
    eval_losses = []
    steps = []
    learning_rates = []
    grad_norms = []
    epochs = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 提取训练配置
        config = extract_training_config(content)
        print(f"检测到的训练配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 匹配训练loss: {'loss': 5.9264, 'grad_norm': 1.336, 'learning_rate': 5e-07, 'epoch': 0.01}
        train_pattern = r"{'loss':\s*([\d.]+),\s*'grad_norm':\s*([\d.]+),\s*'learning_rate':\s*([\d.e-]+),\s*'epoch':\s*([\d.]+)}"
        train_matches = re.findall(train_pattern, content)
        
        # 动态确定步数间隔
        logging_steps = config.get('logging_steps', 5)
        
        for i, match in enumerate(train_matches):
            loss, grad_norm, lr, epoch = match
            train_losses.append(float(loss))
            grad_norms.append(float(grad_norm))
            learning_rates.append(float(lr))
            epochs.append(float(epoch))
            steps.append((i + 1) * logging_steps)
        
        # 匹配验证loss - 先找到所有包含eval_loss的行
        eval_lines = [line for line in content.split('\n') if 'eval_loss' in line]
        print(f"找到 {len(eval_lines)} 行包含 eval_loss 的内容")
        
        # 更强的正则表达式，支持科学记数法和长数字
        eval_pattern = r"'eval_loss':\s*([\d.e-]+)"
        eval_matches = re.findall(eval_pattern, content)
        
        print(f"正则匹配结果: {eval_matches}")
        
        # 如果正则匹配不完整，尝试手动解析
        if len(eval_matches) != len(eval_lines):
            print("正则匹配不完整，尝试手动解析...")
            eval_matches = []
            for line in eval_lines:
                # 查找 'eval_loss': 后面的数字
                match = re.search(r"'eval_loss':\s*([\d.e-]+)", line)
                if match:
                    eval_matches.append(match.group(1))
                    print(f"手动解析: {match.group(1)}")
        
        # 根据实际找到的验证loss数量生成对应的步数
        eval_steps_interval = config.get('eval_steps', 50)
        eval_steps = []
        
        for i, match in enumerate(eval_matches):
            try:
                loss_value = float(match)
                eval_losses.append(loss_value)
                # 生成对应的步数：50, 100, 150, 200...
                eval_steps.append((i + 1) * eval_steps_interval)
                print(f"验证点 {i+1}: Step {(i + 1) * eval_steps_interval}, Loss {loss_value:.6f}")
            except ValueError:
                print(f"无法解析的loss值: {match}")
                continue
        
        print(f"\n检测到 {len(eval_losses)} 个验证点，对应步数: {eval_steps}")
    
    return {
        'train_loss': train_losses,
        'eval_loss': eval_losses,
        'steps': steps,
        'eval_steps': eval_steps,
        'learning_rate': learning_rates,
        'grad_norm': grad_norms,
        'epoch': epochs,
        'config': config
    }

def save_to_json(data, json_file_path):
    """保存数据到JSON文件"""
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {json_file_path}")

def plot_training_curves(data, save_path=None):
    """绘制训练曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练和验证Loss
    ax1.plot(data['steps'], data['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    if data['eval_loss'] and data['eval_steps']:
        # 确保两个数组长度一致
        eval_steps = data['eval_steps'][:len(data['eval_loss'])]
        eval_loss = data['eval_loss'][:len(data['eval_steps'])]
        ax1.plot(eval_steps, eval_loss, 'ro-', linewidth=2, label='Validation Loss', markersize=8)
        print(f"绘制验证Loss: {len(eval_steps)} 个点")
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 学习率变化
    ax2.plot(data['steps'], data['learning_rate'], 'g-', linewidth=2, alpha=0.8)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. 梯度范数
    ax3.plot(data['steps'], data['grad_norm'], 'orange', linewidth=2, alpha=0.8)
    ax3.set_title('Gradient Norm', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Gradient Norm')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss下降趋势分析
    if len(data['train_loss']) > 10:
        # 计算移动平均
        window = min(5, len(data['train_loss']) // 4)
        moving_avg = pd.Series(data['train_loss']).rolling(window=window).mean()
        ax4.plot(data['steps'], data['train_loss'], 'b-', alpha=0.3, label='Raw Loss')
        ax4.plot(data['steps'], moving_avg, 'r-', linewidth=3, label=f'Moving Average (window={window})')
        ax4.set_title('Loss Trend Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印训练配置
    if 'config' in data:
        print("\n=== 训练配置 ===")
        config = data['config']
        
        # 计算有效批次大小
        batch_size = config.get('per_device_train_batch_size', 'unknown')
        accumulation = config.get('gradient_accumulation_steps', 1)
        effective_batch = batch_size * accumulation if isinstance(batch_size, int) else 'unknown'
        
        print(f"学习率: {config.get('learning_rate', 'unknown')}")
        print(f"总步数: {config.get('max_steps', 'unknown')}")
        print(f"Warmup步数: {config.get('warmup_steps', 'unknown')}")
        print(f"批次大小: {batch_size} × {accumulation} = {effective_batch}")
        print(f"验证间隔: 每{config.get('eval_steps', 'unknown')}步")
        print(f"日志间隔: 每{config.get('logging_steps', 'unknown')}步")
        
        # 计算训练进度
        actual_steps = len(data['train_loss']) * config.get('logging_steps', 5)
        max_steps = config.get('max_steps', actual_steps)
        progress = (actual_steps / max_steps * 100) if max_steps > 0 else 0
        print(f"训练进度: {actual_steps}/{max_steps} 步 ({progress:.1f}%)")
    
    # 打印统计信息
    print("\n=== 训练统计 ===")
    print(f"总训练步数: {len(data['train_loss'])}")
    print(f"初始Loss: {data['train_loss'][0]:.4f}")
    print(f"最终Loss: {data['train_loss'][-1]:.4f}")
    print(f"Loss下降: {(data['train_loss'][0] - data['train_loss'][-1]):.4f}")
    print(f"下降百分比: {((data['train_loss'][0] - data['train_loss'][-1]) / data['train_loss'][0] * 100):.2f}%")
    
    if data['eval_loss'] and len(data['eval_loss']) >= 2:
        print(f"\n=== 验证集统计 ===")
        print(f"初始验证Loss: {data['eval_loss'][0]:.4f}")
        print(f"最终验证Loss: {data['eval_loss'][-1]:.4f}")
        print(f"验证Loss下降: {(data['eval_loss'][0] - data['eval_loss'][-1]):.4f}")
    
    print(f"\n=== 学习率范围 ===")
    print(f"最小学习率: {min(data['learning_rate']):.2e}")
    print(f"最大学习率: {max(data['learning_rate']):.2e}")
    
    print(f"\n=== 梯度范数 ===")
    print(f"平均梯度范数: {sum(data['grad_norm'])/len(data['grad_norm']):.4f}")
    print(f"最大梯度范数: {max(data['grad_norm']):.4f}")

def main():
    """主函数"""
    # 文件路径
    log_file = "log.txt"
    json_file = "training_log.json"
    plot_file = "training_curves.png"
    
    # 检查文件是否存在
    if not Path(log_file).exists():
        print(f"错误: 找不到日志文件 {log_file}")
        print("请确保文件存在并且路径正确")
        return
    
    print("开始处理训练日志...")
    print("=" * 50)
    
    # 提取数据
    data = extract_loss_from_log(log_file)
    
    if not data['train_loss']:
        print("错误: 未找到训练loss数据，请检查日志格式")
        return
    
    # 保存JSON
    save_to_json(data, json_file)
    
    # 绘制图表
    plot_training_curves(data, plot_file)
    
    print("\n" + "=" * 50)
    print("处理完成！")

if __name__ == "__main__":
    main()