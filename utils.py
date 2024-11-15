"""
utils.py

这个文件包含了一些常用的深度学习工具函数，可以在深度学习任务中反复使用，
以减少重复代码，提高开发效率。它涵盖了随机种子设置、模型保存与加载、获取优化器和学习率调度器、日志记录、模型冻结等通用功能。

文件名：utils.py

功能概述：
1. 设置随机种子确保实验的可重复性。
2. 保存和加载模型检查点。
3. 获取设备、优化器、调度器等。
4. 模型参数统计与层的冻结/解冻。
5. 计算准确率、设置日志记录等。
6. 其他辅助函数，如学习率调整、早停机制等。
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import logging


def set_seed(seed: int):
    """
    设置随机种子，以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    保存模型的检查点。
    参数：
    - model: 需要保存的模型
    - optimizer: 优化器的状态
    - epoch: 当前训练的轮次
    - loss: 当前的损失值
    - save_path: 保存检查点的路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    logging.info(f'Checkpoint saved at {save_path}')


def load_checkpoint(model, optimizer, load_path, device='cpu'):
    """
    加载模型的检查点。
    参数：
    - model: 需要加载权重的模型
    - optimizer: 优化器，用于加载优化器状态
    - load_path: 检查点的路径
    - device: 加载到的设备（默认为CPU）
    返回值：加载后的模型，优化器，训练的轮次，损失值
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f'Checkpoint loaded from {load_path} at epoch {epoch}')
    return model, optimizer, epoch, loss


def get_device():
    """
    获取可用的设备（如果有CUDA可用则返回CUDA，否则返回CPU）。
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    """
    计算模型中可训练参数的数量。
    参数：
    - model: 需要计算参数的模型
    返回值：模型中可训练参数的总数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model, optimizer_name='adam', learning_rate=0.001):
    """
    获取模型的优化器。
    参数：
    - model: 需要优化的模型
    - optimizer_name: 优化器名称（'adam' 或 'sgd'）
    - learning_rate: 学习率
    返回值：优化器实例
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')


def get_scheduler(optimizer, scheduler_name='step', step_size=7, gamma=0.1):
    """
    获取学习率调度器。
    参数：
    - optimizer: 优化器
    - scheduler_name: 调度器名称（'step' 或 'plateau'）
    - step_size: 学习率下降的步长（仅对 StepLR 有效）
    - gamma: 学习率下降的倍率
    返回值：学习率调度器实例
    """
    if scheduler_name.lower() == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        raise ValueError(f'Unsupported scheduler: {scheduler_name}')


def save_model_summary(model, input_size, save_path):
    """
    保存模型的摘要信息到一个文本文件中。
    参数：
    - model: 需要生成摘要的模型
    - input_size: 模型输入的尺寸
    - save_path: 保存摘要的路径
    """
    from torchsummary import summary
    summary_str = str(summary(model, input_size))
    with open(save_path, 'w') as f:
        f.write(summary_str)
    logging.info(f'Model summary saved at {save_path}')


def accuracy(predictions, labels):
    """
    计算预测的准确率。
    参数：
    - predictions: 模型的预测输出
    - labels: 真实标签
    返回值：准确率（正确预测的比例）
    """
    _, preds = torch.max(predictions, 1)
    return torch.sum(preds == labels).item() / len(labels)


def setup_logging(log_file='training.log'):
    """
    设置日志记录，包括文件和控制台输出。
    参数：
    - log_file: 日志文件的路径
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])


def freeze_layers(model, layer_names=[]):
    """
    冻结模型中指定的层，使其参数不参与训练。
    参数：
    - model: 需要冻结层的模型
    - layer_names: 需要冻结的层的名称列表
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def unfreeze_layers(model, layer_names=[]):
    """
    解冻模型中指定的层，使其参数重新参与训练。
    参数：
    - model: 需要解冻层的模型
    - layer_names: 需要解冻的层的名称列表
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True


def save_predictions(predictions, labels, save_path):
    """
    保存模型的预测结果和真实标签到一个CSV文件中。
    参数：
    - predictions: 模型的预测结果
    - labels: 真实标签
    - save_path: 保存CSV文件的路径
    """
    import pandas as pd
    df = pd.DataFrame({'predictions': predictions.tolist(), 'labels': labels.tolist()})
    df.to_csv(save_path, index=False)
    logging.info(f'Predictions saved at {save_path}')


def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_rate=0.1, lr_decay_epochs=30):
    """
    调整学习率，通常用于训练中动态地减小学习率。
    参数：
    - optimizer: 优化器
    - epoch: 当前的训练轮次
    - initial_lr: 初始学习率
    - lr_decay_rate: 学习率衰减倍率
    - lr_decay_epochs: 每多少个轮次衰减一次学习率
    """
    new_lr = initial_lr * (lr_decay_rate ** (epoch // lr_decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    logging.info(f'Learning rate adjusted to {new_lr}')


def early_stopping(val_loss, best_loss, patience_counter, patience=10):
    """
    早停机制，用于在验证集性能不再提高时提前停止训练。
    参数：
    - val_loss: 当前验证集的损失
    - best_loss: 当前最好的验证集损失
    - patience_counter: 忍耐计数器，记录多少轮次没有提高
    - patience: 忍耐的最大轮次数
    返回值：更新后的最佳损失和计数器，以及是否应该停止训练的布尔值
    """
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    stop_training = patience_counter >= patience
    return best_loss, patience_counter, stop_training


def visualize_training(history, save_path='training_plot.png'):
    """
    可视化训练过程，包括损失和准确率曲线。
    参数：
    - history: 训练历史记录，包括损失和准确率
    - save_path: 保存图像的路径
    """
    import matplotlib.pyplot as plt
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path)
    logging.info(f'Training plot saved at {save_path}')
