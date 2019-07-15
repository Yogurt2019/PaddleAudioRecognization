import numpy as np
import paddle
import librosa
from CNN import *
import os
import warnings
import random
from reader import *
import mobile_net_v1

warnings.filterwarnings(action='ignore')


TRAINING_TIMES = 10
LEARNING_RATE = 0.001


def training_model(training_times):
    for pass_id in range(training_times):
        # 进行训练
        for batch_id, data in enumerate(train_reader()):
            train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, acc])
            if batch_id % 5 == 0:
                print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                      (pass_id + 1, batch_id + 1, train_cost[0], train_acc[0]))

        # 进行测试
        test_accs = []
        test_costs = []
        for batch_id, data in enumerate(test_reader()):
            test_cost, test_acc = exe.run(program=test_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost, acc])
            test_accs.append(test_acc[0])
            test_costs.append(test_cost[0])
        # 求测试结果的平均值
        test_cost = (sum(test_costs) / len(test_costs))
        test_acc = (sum(test_accs) / len(test_accs))
        print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id + 1, test_cost, test_acc))


wav = fluid.layers.data(name='wav', shape=[1, MFCC_LEN, AUDIO_LEN], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


# 获取分类器
model = mobile_net_v1.net(wav)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=LEARNING_RATE, regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)

# 获取wav数据
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train(), buf_size=20000),
    batch_size=128)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        test(), buf_size=20000),
    batch_size=32)

# test_reader = paddle.batch(test(), batch_size=4)

# 定义一个使用CPU的执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[wav, label])

# 开始训练和测试
training_model(TRAINING_TIMES)

# 保存预测模型
save_path = 'infer_model_inference'
save_path2 = 'persistable_model'
# 创建保持模型文件目录
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    os.remove(save_path)
    os.makedirs(save_path)
# 保存预测模型
fluid.io.save_inference_model(save_path, feeded_var_names=[wav.name], target_vars=[model], executor=exe)
fluid.io.save_persistables(executor=exe, dirname=save_path2)

