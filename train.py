import glob
import os
import random
from itertools import chain
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.vit import ViT  # 引入ViT模型

# 设置训练参数
batch_size = 64
epochs = 20
lr = 4e-5
gamma = 0.7
seed = 42

# 设置随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# 定义设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 数据路径
train_dir = 'data/train'
test_dir = 'data/test'

# 获取训练和测试数据列表
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

# 提取标签
labels = [path.split('/')[-1].split('.')[0] for path in train_list]

# 可视化部分训练数据
random_idx = np.random.randint(1, len(train_list), size=9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, ax in enumerate(axes.ravel()):
    img = Image.open(train_list[random_idx[idx]])
    ax.set_title(labels[random_idx[idx]])
    ax.imshow(img)

# 划分训练和验证数据集
train_list, valid_list = train_test_split(
    train_list, test_size=0.2, stratify=labels, random_state=seed
)
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

# 定义数据增强和预处理
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 自定义数据集类
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label

# 加载数据集
train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=val_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

# 定义数据加载器
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 初始化ViT模型
model = ViT(
    dim=128,
    image_size=224,
    patch_size=16,
    num_classes=2,
    channels=3,
    depth=12,
    heads=8,
    mlp_dim=128,
).to(device)

# 定义损失函数、优化器和学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# 初始化最佳验证准确率
best_val_acc = 0.0
save_path = "best_model.pth"  # 模型保存路径

# 训练和验证循环
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    step = 0  # 记录步数

    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        data, label = data.to(device), label.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        step += 1
        # 每 100 步输出一次日志
        if step % 100 == 0:
            print(
                f"Epoch: {epoch + 1}, Step: {step}, "
                f"Step Loss: {loss.item():.4f}, Step Acc: {acc.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0

        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)

            # 验证集前向传播
            val_output = model(data)
            val_loss = criterion(val_output, label)

            # 计算验证准确率
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    # 输出每轮结果
    print(
        f"Epoch : {epoch + 1} - "
        f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - "
        f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    # 如果验证集准确率更高，则保存模型
    if epoch_val_accuracy > best_val_acc:
        best_val_acc = epoch_val_accuracy
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with val_acc: {best_val_acc:.4f}")

