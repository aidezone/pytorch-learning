
import base
import torch
import torchvision


# 声明模型路径
PATH = './data/AlexNet_animals.pth'

# 读取部分测试数据
dataiter = iter(base.testloader)
images, labels = dataiter.next()

# 实例化网络并加载训练完成的模型
# net = base.MyCustomNet()
print(len(base.classes))
net = base.AlexNet(len(base.classes))
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
    if total_pred[classname] > 0:
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    else:
        print(f'Accuracy for class: {classname:5s} is NaN')
    