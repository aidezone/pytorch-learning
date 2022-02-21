
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.image as plimg
import json


# 定义图片通道数、宽、高
CHANNEL = 3
WIDTH = 32
HEIGHT = 32
 
data = [] #反序列化的numpy数据
labels = [] #图片标注
img = [] #图像数据

# 图片标注映射表
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 读取数据集文件
for i in range(5):
    with open("./data/cifar-10-batches-py/data_batch_"+ str(i+1),mode='rb') as file:
    #数据集在当脚本前文件夹下
        data_dict = pickle.load(file, encoding='bytes')
        data+= list(data_dict[b'data'])
        labels+= list(data_dict[b'labels'])

# 将numpy数据转换成图像通道数组
img =  np.reshape(data,[-1,CHANNEL, WIDTH, HEIGHT])


#代码创建文件夹，也可以自行创建 
data_path = "./data/pic"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# 拿出5000张图片作为测试集
fo = open("./data/test_data_meta.jsonl", "w")
for i in range(0, 5000):
    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))
 
    name = "img-" + str(i) +"-"+ classification[labels[i]]+ ".png"

    fo.write(json.dumps({'filename': name, 'label': labels[i]}) + "\n")
    rgb.save(data_path + '/' + name, "PNG")
fo.close()

# 剩余图片作为训练集
fo = open("./data/train_data_meta.jsonl", "w")
for i in range(5000, len(img)):
    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))
 
    name = "img-" + str(i) +"-"+ classification[labels[i]]+ ".png"

    fo.write(json.dumps({'filename': name, 'label': labels[i]}) + "\n")
    rgb.save(data_path + '/' + name, "PNG")
fo.close()

# 写入标注映射文件
fo = open("./data/label_dict.json", "w")
fo.write(json.dumps(classification))
fo.close()

