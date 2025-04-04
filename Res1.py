import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os
from multiprocessing import freeze_support
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# ---------------------- 数据预处理 ----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ---------------------- 模型加载 ----------------------
def create_model(num_classes, freeze_backbone=True):  # 修改冻结策略
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        # 只训练最后两个stage + 分类头
        for name, param in model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    """集成之前的可视化方案"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy')
    plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
# ---------------------- 主程序 ----------------------
def main():
    freeze_support()
    
    # 启用cudNN优化（重要！）
    torch.backends.cudnn.benchmark = True   # 自动寻找最优卷积算法
    torch.backends.cudnn.enabled = True     # 启用加速库
    torch.set_float32_matmul_precision('high')  # Ampere+架构优化
    
    # 数据集路径
    data_dir = r'C:\Users\Lenovo\Desktop\人工神经网络\data\StanfordDogs_split'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 优化后的数据加载参数
    BATCH_SIZE = 128
    NUM_WORKERS = min(os.cpu_count(), 8)  # 自适应worker数量
    val_accuracies = []
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    # 优化后的DataLoader配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,        # 预加载批次
        drop_last=True            # 避免最后不完整批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=max(NUM_WORKERS//2, 2),  # 验证集减少workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    # 训练参数
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = len(train_dataset.classes)
    EPOCHS = 20
    LR = 1e-4
    HEAD_LR = 1e-3

    # 初始化模型
    model = create_model(NUM_CLASSES).to(DEVICE)
    
    # 分层参数设置
    head_params = list(model.fc.parameters())
    backbone_params = [
        p for name, p in model.named_parameters() 
        if "fc" not in name and p.requires_grad
    ]

    optimizer_params = [
        {"params": head_params, "lr": HEAD_LR},
        {"params": backbone_params, "lr": LR}
    ]

    # 优化器配置
    optimizer = optim.AdamW(optimizer_params, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(init_scale=1024, growth_interval=2000)  # 改进的梯度缩放
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        # 训练阶段添加准确率计算
        train_loss = 0.0
        correct_train = 0  # 新增训练正确计数
        total_train = 0
        # 使用CUDA流并行
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for images, labels in train_loader:
                # 异步数据传输
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                

                # 混合精度训练（bfloat16需要Ampere+架构）
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * images.size(0)
                 # 训练准确率计算
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)       
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        # 学习率调整
        scheduler.step()
        
        # 修改验证阶段的指标记录
        train_loss /= len(train_dataset)
        train_acc = correct_train / total_train  # 计算训练准确率
        val_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)
        
        # 记录所有指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_finetune_model.pth")

        val_accuracies.append(val_acc)
        epoch_time = time.time() - start_time
        
        # 打印带吞吐量的信息
        samples_sec = len(train_dataset) / epoch_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Throughput: {samples_sec:.0f} samples/s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 60)
    plot_metrics(train_losses, train_accs, val_losses, val_accs)  # 替换原来的plt.plot(val_accuracies)

if __name__ == '__main__':
    main()

