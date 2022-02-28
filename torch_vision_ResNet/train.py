# 参考资料：
# https://www.cnblogs.com/zhengbiqing/p/10432169.html
# http://blog.itpub.net/31555081/viewspace-2705514/
# https://zhuanlan.zhihu.com/p/225597229
# https://www.cnblogs.com/wzyuan/p/9880342.html
# https://blog.csdn.net/wuzhuoxi7116/article/details/106309943
# 重点：https://blog.csdn.net/qq_43620727/article/details/122673962


# 开始真正的模型训练
import base


import torch
import torch.nn as nn
import torchvision.models as models
import datetime


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

# 实例化网络
net = models.resnet34(pretrained=True)
# 2.提取fc层中固定的参数
fc_features = net.fc.in_features
# 3.修改输出的类别为10
net.fc = nn.Linear(fc_features, len(base.classes))
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





