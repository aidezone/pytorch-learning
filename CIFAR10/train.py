import base


# 开始真正的模型训练
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


