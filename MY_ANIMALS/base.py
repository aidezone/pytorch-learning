
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
        return image, label

# ################################################################################

# # 使用Loader加载自定义数据集
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import json

# transform = transforms.Compose(
#     [transforms.Resize((32, 32)),  #将图片resize到与网络输入要求匹配的32*32的大小。
#      transforms.ToTensor(),
#      # transforms.Lambda(lambda x: x.repeat(3,1,1)),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# # 图片地址
# img_dir = "../Image-Downloader/download_images"
# # label映射文件
# label_dict = "./data/label_dict.json"
# # 加载图片时的batch_size
# batch_size = 4
# # 预定义label变量
# classes = []

# # 遍历文件夹，按顺序给出labeldict
# for item in os.listdir(img_dir):
#     if os.path.isdir(os.path.join(img_dir, item)):
#         classes.append(item)

# print(f'classes: {classes}')

# # # 实例化训练集、测试
# training_data = CustomImageDataset(classes, 0, 80, img_dir, transform=transform)
# test_data = CustomImageDataset(classes, 80, 2000, img_dir, transform=transform)

# print('training_data len: ' + str(len(training_data)))
# print('test_data len: ' + str(len(test_data)))


# # 实例化加载器
# trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)



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

# 定义AlexNet
# 源自：https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/alexnet.py
# 实际测试后发现这段定义不可以直接使用，故放弃。
# class AlexNet(nn.Module):
#     """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.
#     The input for AlexNet is a 224x224 RGB image.
#     Args:
#         num_classes (int): number of classes for classification.
#             The default value is -1, which uses the backbone as
#             a feature extractor without the top classifier.
#     """

#     def __init__(self, num_classes=-1):
#         super(AlexNet, self).__init__()
#         self.num_classes = num_classes
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         if self.num_classes > 0:
#             self.classifier = nn.Sequential(
#                 nn.Dropout(),
#                 nn.Linear(256 * 6 * 6, 4096),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(),
#                 nn.Linear(4096, 4096),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(4096, num_classes),
#             )

#     def forward(self, x):
#         x = self.features(x)
#         if self.num_classes > 0:
#             x = x.view(x.size(0), 256 * 6 * 6)
#             x = self.classifier(x)
#         return (x, )


# 摘自：https://blog.csdn.net/weixin_44023658/article/details/105798326
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):   
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  #打包
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            nn.ReLU(inplace=True), #inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27] kernel_num为原论文一半
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全链接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) #展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #何教授方法


################################################################################

# 重新定义数据加载
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

transform = transforms.Compose(
    [transforms.Resize((224, 224)),  #将图片resize到与Alex网络输入要求匹配的224*224的大小。
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

