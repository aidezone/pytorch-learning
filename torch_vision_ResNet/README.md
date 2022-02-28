[TOC]

# 准备工作目录
```bash
# 获取当前示例代码
git clone https://github.com/aidezone/pytorch-learning.git

# 进入工作目录
cd pytorch-learning

# 创建数据目录
mkdir data
````



# 复用MY_ANIMALS数据

## 使用ImageDownloader

```bash
git clone https://github.com/aidezone/Image-Downloader

````

详细说明：https://github.com/aidezone/Image-Downloader/blob/master/README_zh.md


# 使用torchvision中提供的网络

## 参考资料：
* https://www.cnblogs.com/zhengbiqing/p/10432169.html
* http://blog.itpub.net/31555081/viewspace-2705514/
* https://zhuanlan.zhihu.com/p/225597229
* https://www.cnblogs.com/wzyuan/p/9880342.html
* https://blog.csdn.net/wuzhuoxi7116/article/details/106309943
* 重点：https://blog.csdn.net/qq_43620727/article/details/122673962


## 修改transformer

```python

transform = transforms.Compose(
    [transforms.Resize((800, 600)),  #将图片resize到同样大小，我们使用resnet训练，对于输入大小只要求一致即可，不需要严格限定
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


```



## 训练并测试模型

```bash

python torch_vision_ResNet/train.py

python torch_vision_ResNet/evaluate.py

```
### 定义训练方法
```python

global_train_acc = []
global_test_acc = []
def net_train(net, loss_func, train_data_load, optimizer, epoch, log_interval):
    net.train()
    begin = datetime.datetime.now()
    # 样本总数
    total = len(train_data_load.dataset)
    # 样本批次训练的损失函数值的和
    train_loss = 0
    # 识别正确的样本数
    ok = 0

    for i, data in enumerate(train_data_load, 0):
        img, label, _ = data
        optimizer.zero_grad()

        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

        # 累加损失值和训练样本数
        train_loss += loss.item()
        _, predicted = torch.max(outs.data, 1)
        # 累加识别正确的样本数
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            # 已训练的样本数
            traind_total = (i + 1) * len(label)
            # 准确度
            acc = 100. * ok / traind_total
            # 记录训练准确率以输出变化曲线
            global_train_acc.append(acc)
    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)
```

### 加载torchvision中提供的resnet网络并修改模型实例化代码

```python

import torchvision.models as models

# 实例化网络并加载训练完成的模型
net = models.resnet34(pretrained=True)
# 2.提取fc层中固定的参数
fc_features = net.fc.in_features
# 3.修改输出的类别为10
net.fc = nn.Linear(fc_features, len(base.classes))

```

### 训练并保存模型

```python
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
    net_train(net, criterion, base.trainloader, optimizer, epoch, 20)

print('Finished Training')

PATH = './data/ResNet_animals.pth'
torch.save(net.state_dict(), PATH)

```

### 训练输出
```bash
(pytorch-learning) gaoyuan1@cn0214006387m pytorch-learning % python torch_vision_ResNet/train.py
Using cpu device
classes: ['马', '牛', '狗', '鸡', '猪', '猫', '羊', '鸭', '鹅']
training_data len: 738
test_data len: 79
one epoch spend:  0:04:07.693607
one epoch spend:  0:04:10.253857
one epoch spend:  0:04:03.146926
one epoch spend:  0:04:04.520495
one epoch spend:  0:04:04.810037
Finished Training
```

### 评测输出

```bash
(pytorch-learning) gaoyuan1@cn0214006387m pytorch-learning % python torch_vision_ResNet/evaluate.py
Using cpu device
classes: ['马', '牛', '狗', '鸡', '猪', '猫', '羊', '鸭', '鹅']
training_data len: 738
test_data len: 79
标注结果:  鸡     猫     狗     牛
推理结果:  鸡     猫     狗     牛
Accuracy of the network on the test images: 84 %
Accuracy for class: 马     is 75.0 %
Accuracy for class: 牛     is 66.7 %
Accuracy for class: 狗     is 100.0 %
Accuracy for class: 鸡     is 92.3 %
Accuracy for class: 猪     is 100.0 %
Accuracy for class: 猫     is 83.3 %
Accuracy for class: 羊     is 85.7 %
Accuracy for class: 鸭     is 76.9 %
Accuracy for class: 鹅     is 100.0 %
```
### 启动可视化界面（复用）（show_dataset.py）

```bash
python torch_vision_ResNet/show_dataset.py

```

# 小结

1. 学习了三个示例之后发现loss函数都可以使用CrossEntropyLoss，求教算法工程师后得到答案：单分类算法使用的loss函数大多为CrossEntropyLoss
2. torchvision中提供了多种常见网络的实现，可以import直接使用，具体用法有很多相关文章可以参考。







