import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
import os
import numpy as np
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

# ---------------------- 数据预处理 ----------------------
class EfficientColorJitter(transforms.ColorJitter):
    """优化颜色增强，70%概率跳过增强"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        super().__init__(brightness, contrast, saturation)
        
    def forward(self, img):
        if np.random.rand() > 0.3:
            return img
        return super().forward(img)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    EfficientColorJitter(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),  # 在归一化前进行擦除
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------- DenseNet模型 ----------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return self.pool(x)

class OptimizedDenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=32, block_config=(4, 8, 16, 12), compression=0.5):
        """优化后的配置，减少层数加速训练"""
        super().__init__()
        self.in_channels = 2 * growth_rate
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, self.in_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # 构建DenseBlock和Transition
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, self.in_channels, growth_rate)
            self.features.add_module(f'denseblock_{i+1}', block)
            self.in_channels += num_layers * growth_rate
            
            if i != len(block_config)-1:
                out_channels = int(self.in_channels * compression)
                trans = Transition(self.in_channels, out_channels)
                self.features.add_module(f'transition_{i+1}', trans)
                self.in_channels = out_channels

        # 最终分类层
        self.final_bn = nn.BatchNorm2d(self.in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # 优化后的参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.final_bn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------------------- 训练工具函数 ----------------------
def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves'), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves'), plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.legend()
    
    plt.tight_layout()
    plt.savefig('densenet_metrics.png')
    plt.close()

# ---------------------- 主程序 ----------------------
def main():
    freeze_support()
    
    # 启用cudNN优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # 数据集路径
    data_dir = r'C:\Users\Lenovo\Desktop\人工神经网络\data\StanfordDogs_split'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 优化数据加载参数
    BATCH_SIZE = 128
    NUM_WORKERS = min(os.cpu_count(), 8)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE*2,
        shuffle=False,
        num_workers=NUM_WORKERS//2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 训练参数
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = len(train_dataset.classes)
    EPOCHS = 100
    LR = 4e-4
    GRAD_ACCUM = 2  # 梯度累积步数

    # 初始化模型
    model = OptimizedDenseNet(NUM_CLASSES).to(DEVICE)
    
    # 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 训练记录
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    # 训练循环
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        start_time = time.time()
        model.train()
        
        # 训练阶段
        train_loss = 0.0
        correct_train = 0
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # 混合精度训练
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels) / GRAD_ACCUM
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (i+1) % GRAD_ACCUM == 0 or (i+1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(epoch + i/len(train_loader))
            
            # 统计指标
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        # 计算指标
        train_loss /= len(train_dataset)
        train_acc = correct_train / len(train_dataset)
        val_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)
        
        # 记录历史数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_densenet.pth")
        
        # 打印统计信息
        epoch_time = time.time() - start_time
        throughput = len(train_dataset) / epoch_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Throughput: {throughput:.0f} samples/s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 60)

    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()

