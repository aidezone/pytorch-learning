
# 训练原文：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# 自定义数据集原文：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


# Get cpu or gpu device for training.
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 定义数据集
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # 加载标注数据（meta文件）
        self.img_labels = []
        fo = open(annotations_file, "r+")
        for line in fo:
            linestr = line.strip()
            if linestr != "":
                self.img_labels.append(json.loads(line.strip()))
        fo.close()
        # 定义图片目录
        self.img_dir = img_dir
        # 定义transform，当前还不理解此处具体的作用，暂且根据官方示例来写。
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) - 1

    def __getitem__(self, idx):
        # 拼接图片完整路径
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["filename"])
        # 读取一张图片
        image = Image.open(img_path)
        # 读取图片对应的标签
        label = self.img_labels[idx]["label"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

################################################################################

# 使用Loader加载自定义数据集
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 图片地址
img_dir = "./data/pic"
# 训练集meta文件
train_file = "./data/train_data_meta.jsonl"
# 测试集meta文件
test_file = "./data/test_data_meta.jsonl"
# label映射文件
label_dict = "./data/label_dict.json"
# 加载图片时的batch_size
batch_size = 4
# 预定义label变量
classes = []

# 实例化训练集、测试
training_data = CustomImageDataset(train_file, img_dir, transform=transform)
test_data = CustomImageDataset(test_file, img_dir, transform=transform)

# 实例化加载器
trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# 加载labeldict文件
with open(label_dict, 'r', encoding='utf8') as fp:
    classes = json.load(fp)


################################################################################


import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络
class MyCustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
