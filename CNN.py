import paddle.fluid as fluid
import numpy as np

# 定义多层感知器
def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=128, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=128, act='relu')
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2, size=2, act='sigmoid')
    return fc

# 卷积神经网络
def convolutional_neural_network1(input):
    print(np.shape(input))
    # 第一个卷积层，卷积核大小为3*3，一共有16个卷积核
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=16,
                                filter_size=3,
                                stride=1)

    # 第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=(2, 1),
                                pool_type='max')

    bn1 = fluid.layers.batch_norm(input=pool1, name='bn1')

    # 第二个卷积层，卷积核大小为3*3，一共有64个卷积核
    conv2 = fluid.layers.conv2d(input=bn1,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    # 第二个池化层，池化大小为2*2，步长为1，最大池化
    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max')

    bn2 = fluid.layers.batch_norm(input=pool2, name='bn2')

    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc1 = fluid.layers.fc(input=bn2, size=64, act='relu')
    fc2 = fluid.layers.fc(input=fc1, size=2, act='softmax')

    return fc2

# 卷积神经网络2
def convolutional_neural_network(input):
    print(np.shape(input))

    # 第一个卷积层，卷积核大小为3*3，一共有16个卷积核
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=16,
                                filter_size=3,
                                stride=1)

    # 第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=(2, 1),
                                pool_stride=1,
                                pool_type='max')

    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool1, size=2, act='softmax')
    return fc
