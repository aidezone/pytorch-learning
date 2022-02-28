import base

# 验证加载器
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
