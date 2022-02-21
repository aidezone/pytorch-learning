# 准备工作目录
```bash
# 获取当前示例代码
git clone https://github.com/aidezone/pytorch-learning.git

# 进入工作目录
cd pytorch-learning

# 创建数据目录
mkdir data
````



# 准备数据集 （make_data.py）

## 获取开源数据集
*下载地址*

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar-10-python.tar.gz

````

*解压数据集*


```bash
tar zxvf ./data/cifar-10-python.tar.gz -C ./data

````

## 生成数据图片和meta


```bash

# 查看示例文件
cat ./CIFAR10/make_data.py

# 运行示例 
python3 ./CIFAR10/make_data.py

```

> 还原出来的图片会存放于 ./data/pic
> 标注信息分别使用 ./data/test_data_meta.jsonl ./data/train_data_meta.jsonl存放
> label标签映射存放于 ./data/label_dict.json

---

## 为什么要还原数据集？

学习模型训练最终要达成的效果是：可以把自己采集回来的图片，通过标注工具生成可以用于训练的数据。所以我们需要理解如何构建一个自定义的数据集。
官方的教程里开源数据集都是一些序列化之后的格式，并且可以通过官方的库来加载，框架内部封装了各种格式的解析，我们需要把这些封装打开来学习。
也可以使用业界规范的标注工具产出一些标准的数据格式如COCO、YOLO等。这部分内容在当前阶段不是重点，会放在后面的可视化相关部分重点学习。


# 定义模型训练需要的内容 （base.py）

> 训练原文：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
> 自定义数据集原文：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


## 定义数据集

```python
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

```


## 使用Loader加载自定义数据集

```python
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
```


## 定义网络结构
```python

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
```

## 初始化网络以及优化器
```python

# 实例化网络
net = MyCustomNet()

# 定义数据集的训练迭代轮次
max_epoch = 5

# 定义学习率
learning_rate = 0.001

# 定义loss函数、定义优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
```

# 开始正式的训练工作

1. 验证定义的内容
2. 训练一个模型
3. 

## 验证加载器工作是否正常
```python
import base

import matplotlib.pyplot as plt
import numpy as np

# 定义一个pyplot图片查看器
def imshow(img):
    # print(img)
    # 此处还原的图片是一个tensor，每个像素点中包含负数，需要计算得到0~1之间的每个像素点值，然后将tensor还原为numpy数组
    img = (img / 2 + 0.5).numpy()    # unnormalize
    # print(img)
    # 此处进行矩阵转换，具体作用貌似是把numpy数组转换成可以用于pyplot展示图片使用的数组，不是特别理解，后续慢慢深入研究
    # 详见官方文档：https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    img = np.transpose(img, (1, 2, 0))
    # print(img)
    plt.imshow(img)
    plt.show()

# 展示一张训练集图片，以验证加载器是否正常运行
train_features, train_labels = next(iter(base.trainloader))
img = train_features[0].squeeze()
label = base.classes[train_labels[0]]
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Label: {label}")
imshow(img)

```


## 开始训练
```python
import base

import torch
import torch.nn as nn

# 实例化网络
net = base.MyCustomNet()

# 定义数据集的训练迭代轮次
max_epoch = 5

# 定义学习率
learning_rate = 0.001

# 定义loss函数、定义优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# 进行模型训练
for epoch in range(max_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(base.trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './data/cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

## 模型效果评测

```python

import base
import torch
import torchvision


# 声明模型路径
PATH = './data/cifar_net.pth'

# 读取部分测试数据
dataiter = iter(base.testloader)
images, labels = dataiter.next()

# 实例化网络并加载训练完成的模型
net = base.MyCustomNet()
net.load_state_dict(torch.load(PATH))

# 得到模型推理tensor
outputs = net(images)

# 得到推理结果
_, predicted = torch.max(outputs, 1)

# 显示一部分图片 标注结果、推理结果
print('标注结果: ', ' '.join(f'{base.classes[labels[j]]:5s}' for j in range(4)))
print('推理结果: ', ' '.join(f'{base.classes[predicted[j]]:5s}' for j in range(4)))

# base.imshow(torchvision.utils.make_grid(images))


# 计算全量数据的精度
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in base.testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')


# 按不同的分类计算精度
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in base.classes}
total_pred = {classname: 0 for classname in base.classes}
# again no gradients needed
with torch.no_grad():
    for data in base.testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[base.classes[label]] += 1
            total_pred[base.classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

```

