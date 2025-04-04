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

# ---------------------- 可视化函数 ----------------------
def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    """集成双曲线可视化方案"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线子图
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线子图
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy')
    plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('densenet_training_metrics.png')
    plt.close()

# ---------------------- 模型加载 ----------------------
def create_model(num_classes, freeze_backbone=False):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# ---------------------- 主程序 ----------------------
def main():
    freeze_support()
    
    # 数据集路径
    data_dir = r'C:\Users\Lenovo\Desktop\人工神经网络\data\StanfordDogs_split'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 数据加载参数
    BATCH_SIZE = 64
    NUM_WORKERS = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # 训练参数
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = len(train_dataset.classes)
    EPOCHS = 30
    LR = 1e-4
    HEAD_LR = 1e-3

    # 初始化模型
    model = create_model(NUM_CLASSES, freeze_backbone=False).to(DEVICE)

    # 分层参数设置
    head_params = list(model.classifier.parameters())
    backbone_params = [
        p for name, p in model.named_parameters() 
        if "classifier" not in name and p.requires_grad
    ]

    optimizer_params = [
        {"params": head_params, "lr": HEAD_LR},
        {"params": backbone_params, "lr": LR}
    ]

    # 优化器配置
    optimizer = optim.AdamW(optimizer_params, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    # 训练记录初始化
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    # 训练循环
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计训练指标
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算指标
        train_loss = train_loss / len(train_dataset)
        train_acc = correct_train / total_train
        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / len(val_dataset)
        
        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "best_densenet_model.pth")

        # 打印信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print("-" * 60)

    # 绘制综合曲线
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
