import base


# 开始真正的模型训练
import torch
import torch.nn as nn

# 实例化网络
net = base.AlexNet(len(base.classes))
# X = torch.rand(4, 28, 28)
# print(X)
# net(X)
# 定义数据集的训练迭代轮次
max_epoch = 5

# 定义学习率
learning_rate = 0.001

# 定义loss函数、定义优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#优化器 这里用Adam
optimizer = optim.Adam(net.parameters(), lr=0.0002)
# 进行模型训练
for epoch in range(max_epoch):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(base.trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data
        # print(inputs)
        # print(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './data/AlexNet_animals.pth'
torch.save(net.state_dict(), PATH)


