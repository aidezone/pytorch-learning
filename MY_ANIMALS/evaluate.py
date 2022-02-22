
import base
import torch
import torchvision
import json

# 声明模型路径
PATH = './data/AlexNet_animals.pth'

# 读取部分测试数据
dataiter = iter(base.testloader)
images, labels, image_path = dataiter.next()

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
prediction_file = "./data/my_animals_inferance.jsonl"

# 拿出5000张图片作为测试集
fo = open(prediction_file, "w")
# 输出推理结果
with torch.no_grad():
    for data in base.testloader:
        images, labels, image_path = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction, image in zip(labels, predictions, image_path):
            fo.write(json.dumps({
                'image_path': image, 
                'label': int(label), 
                'prediction': int(prediction),
                'label_name': base.classes[label], 
                'prediction_name': base.classes[prediction],
                }) + "\n")
fo.close()
   


# 计算全量数据的精度
correct = 0
total = 0
fo = open(prediction_file, "r+")
for line in fo:
    linestr = line.strip()
    if linestr == "":
        continue
    lineObj = json.loads(linestr)
    total += 1
    if lineObj["prediction"] == lineObj["label"]:
        correct += 1
fo.close()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')


# 按不同的分类计算精度
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in base.classes}
total_pred = {classname: 0 for classname in base.classes}
# again no gradients needed
fo = open(prediction_file, "r+")
for line in fo:
    linestr = line.strip()
    if linestr == "":
        continue
    lineObj = json.loads(linestr)
    if lineObj["prediction"] == lineObj["label"]:
        correct_pred[lineObj["prediction_name"]] += 1
    total_pred[lineObj["prediction_name"]] += 1
fo.close()

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    if total_pred[classname] > 0:
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    else:
        print(f'Accuracy for class: {classname:5s} is NaN')
    