import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import time
import os
import numpy as np
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

# ---------------------- 数据预处理优化 ----------------------
class EfficientColorJitter(transforms.ColorJitter):
    """优化后的颜色增强，减少计算开销"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        super().__init__(brightness=brightness, 
                        contrast=contrast, 
                        saturation=saturation)
        
    def forward(self, img):
        if np.random.rand() > 0.3:  # 70%概率跳过增强
            return img
        return super().forward(img)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    EfficientColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)  # 添加随机擦除
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------- 模型定义 ----------------------
class OptimizedResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super().__init__()
        self.expansion = 2
        mid_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3,
            stride=stride, padding=1, groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.shortcut(x)        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))        
        out += residual
        return self.act(out)

class EfficientResNeXt(nn.Module):
    def __init__(self, num_classes, cardinality=32, blocks=[2, 3, 4, 2]):      
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个阶段
        self.layer1 = self._make_layer(64, 256, blocks[0], stride=1, cardinality=cardinality)
        self.layer2 = self._make_layer(256, 512, blocks[1], stride=2, cardinality=cardinality)
        self.layer3 = self._make_layer(512, 1024, blocks[2], stride=2, cardinality=cardinality)
        self.layer4 = self._make_layer(1024, 2048, blocks[3], stride=2, cardinality=cardinality)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, cardinality):
        layers = []
        layers.append(OptimizedResNeXtBlock(in_channels, out_channels, stride, cardinality))
        for _ in range(1, num_blocks):
            layers.append(OptimizedResNeXtBlock(out_channels, out_channels, 1, cardinality))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
    plt.savefig('training_metrics.png')
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
    BATCH_SIZE = 64
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
        batch_size=BATCH_SIZE*2,  # 验证集使用更大批次
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
    LR = 5e-4
    GRAD_ACCUM = 2  # 梯度累积步数

    # 初始化模型
    model = EfficientResNeXt(NUM_CLASSES).to(DEVICE)
    
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
                
                # 学习率调度
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
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pth")
        
        # 打印统计信息
        epoch_time = time.time() - start_time
        throughput = len(train_dataset) / epoch_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Throughput: {throughput:.0f} samples/s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 60)

    # 绘制曲线
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()


