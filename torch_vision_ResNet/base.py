
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
import random

class CustomImageDataset(Dataset):
    def __init__(self, classes, offset, limit, img_dir, transform=None, target_transform=None):
        self.img_classes = classes
        # 从文件目录中直接加载标注信息，eg：./download_image/猫/0001.jpg   img_dir=./download_image  classes[i]=猫
        self.img_labels = []
        for i in range(0, len(classes)):
            # 拼接分类名称为一个完整的图片文件夹
            path_with_label = os.path.join(img_dir, classes[i])
            # 遍历每个分类文件夹下的图片
            for item in os.listdir(path_with_label):
                self.img_labels.append({
                    'file': os.path.join(classes[i], item), # img_dir之后的图片文件相对路径，eg：猫/0001.jpg
                    'label_index': i,
                    'label_name': classes[i],
                    })
        # 按输入的数量预期拆分数组
        random.shuffle(self.img_labels)
        self.img_labels = self.img_labels[offset:(offset+limit)]
        # 定义图片目录
        self.img_dir = img_dir
        # 定义transform，从后续的实验看出此部分是定义图片转换处理的，例如：toTensor、Resize
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) - 1

    def __getitem__(self, idx):
        # print(self.img_labels[idx])
        # 拼接图片完整路径
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["file"])
        # print(img_path)
        # 读取一张图片,抓取失败的图片可能是单通道的，此处强转一下
        image = Image.open(img_path).convert("RGB")
        # channels = len(image.split())
        # print(channels)
        # 读取图片对应的标签
        label = self.img_labels[idx]["label_index"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path


################################################################################

# 重新定义数据加载
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

transform = transforms.Compose(
    [transforms.Resize((800, 600)),  #将图片resize到同样大小，我们使用resnet训练，对于输入大小只要求一致即可，不需要严格限定
     transforms.ToTensor(),
     # transforms.Lambda(lambda x: x.repeat(3,1,1)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 图片地址
img_dir = "../Image-Downloader/download_images"
# label映射文件
label_dict = "./data/label_dict.json"
# 加载图片时的batch_size
batch_size = 4
# 预定义label变量
classes = []

# 遍历文件夹，按顺序给出labeldict
for item in os.listdir(img_dir):
    if os.path.isdir(os.path.join(img_dir, item)):
        classes.append(item)

print(f'classes: {classes}')

# # 实例化训练集、测试
test_data = CustomImageDataset(classes, 0, 80, img_dir, transform=transform)
training_data = CustomImageDataset(classes, 80, 2000, img_dir, transform=transform)

print('training_data len: ' + str(len(training_data)))
print('test_data len: ' + str(len(test_data)))


# 实例化加载器
trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

