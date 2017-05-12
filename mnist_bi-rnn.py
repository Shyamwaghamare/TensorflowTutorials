# Bidirectional LSTM classifier

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(train_dir="C:/GitHub/TensorflowTutorials/data", one_hot=True)

# 学习率
learning_rate = 0.01
# 最大样本数
max_samples = 400000
batch_size = 128
# 每间隔10次训练展示一次训练情况
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 256
n_classes =10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 因为双向 LSTM ，有 forward 和 backward 的两个 lstm 的 cell，
# 所以 weights 的参数也翻倍，变为 2*n_hidden
weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BBiRNN(x, weights, biases):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_fw_cell,
        lstm_bw_cell,
        x,
        dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases

pred = BBiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 定义优化器为 Adam
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 使用argmax 得到模型预测的类别，使用 tf.equal 判断是否预测正确，使用 tf.reduce_mean 求得平均准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y:batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", minibatch loss =  " + \
                "{:.6f}".format(loss) + ", training accuracy = " + \
                "{:.5f}".format(acc))
        step += 1
    print("optimization finished!")

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]

    print("test accuracy:",
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

"""
......
Iter 398080, minibatch loss =  0.036608, training accuracy = 0.99219
Iter 399360, minibatch loss =  0.009305, training accuracy = 1.00000
optimization finished!
test accuracy: 0.9723
"""

