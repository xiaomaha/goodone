# TF之LiR：基于tensorflow实现手写数字图片识别准确率

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist)

# 设置超参数
lr = 0.001  # 学习率
training_iters = 100  # 训练次数
batch_size = 128  # 每轮训练数据的大小，如果一次训练5000张图片，电脑会卡死，分批次训练会更好
display_step = 1

# tf Graph的输入
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 设置权重和偏置
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 设定运行模式
pred = tf.nn.softmax(tf.matmul(x, w) + b)  #
# 设置cost function为cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# GD算法
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# 初始化权重
init = tf.global_variables_initializer()
# 开始训练
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_iters):  # 输入所有训练数据
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):  # 遍历每个batch
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})  # 把每个batch数据放进去训练
            avg_cost == c / total_batch
        if (epoch + 1) % display_step == 0:  # 显示每次迭代日志
            print("迭代次数Epoch:", "%04d" % (epoch + 1), "下降值cost=", "{:.9f}".format(avg_cost))
    print("Optimizer Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.equal_mean(tf.cast(correct_prediction), tf.float32)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    #print("Accuracy:", accuracy_eval({x: mnist.test.image[:3000], y: mnist}))