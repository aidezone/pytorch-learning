# 安装训练环境

## 建议先安装一个Conda或者Docker

为了学习工作时多个不同依赖环境的管理，建议安装一个Conda，参考：https://docs.conda.io/en/latest/miniconda.html#installing
如果您对docker非常熟悉，也可以直接用docker，参考：https://hub.docker.com/r/bitnami/pytorch


## 以conda为例创建开发环境

参考原文：https://pytorch.org/get-started/locally/#mac-anaconda

```bash
# 创建一个新的conda环境(一定要注意带python=3.7，版本太高容易出现pip库的版本依赖问题)
conda create -n pytorch-learning python=3.7

# 进入新创建的环境
conda activate pytorch-learning



# 安装pytorch、torchvisin、numpy、matplotlib
conda install pytorch torchvision numpy matplotlib


# 当然也可以使用pip安装
# 检查一下pip是否在当前新创建的conda环境中
which pip

# 注意pip中叫torch不叫pytorch。。。
pip install torch torchvision numpy matplotlib

# 如果速度慢可以使用国内源，添加参数：-i https://pypi.tuna.tsinghua.edu.cn/simple
# 清华源镜像站： https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  torch torchvision numpy matplotlib 


````

## 训练一个简单的分类器（CIFAR10）

更多：https://github.com/aidezone/pytorch-learning/blob/main/CIFAR10/README.md