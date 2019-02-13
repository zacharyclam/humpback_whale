#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : train.py
# @Time     : 2019/1/17 17:58 
# @Software : PyCharm
import tensorflow as tf
import os

from utils import readRecords
slim = tf.contrib.slim

with tf.name_scope("imput"):
    train_x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
    train_y = tf.placeholder(tf.float32, shape=[None, 5005])


def model():
     with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.conv2d], stride=1):
          net = slim.conv2d(train_x, 64, [3, 3])
          net = slim.conv2d(net, 128, [3, 3])

          net = slim.conv2d(net, 256, [3, 3])
          net = slim.conv2d(net, 256, [3, 3])
          net = slim.max_pool2d(net, [2, 2])

          net = slim.conv2d(net, 512, [3, 3])
          net = slim.conv2d(net, 512, [3, 3])
          net = slim.max_pool2d(net, [2, 2])

          net = slim.conv2d(net, 512, [3, 3], scope='conv2')
          net = slim.max_pool2d(net, [2, 2])

          net = slim.flatten(net) 
          net = slim.fully_connected(net, 8192, scope='fc7')
          net = slim.dropout(net, 0.5, scope='dropout7')

          net = slim.fully_connected(net, 4096, scope='fc8')
          net = slim.dropout(net, 0.5)
          net = slim.fully_connected(net, 5005, activation_fn=None, scope='fc')

     return net

learning_rate = 0.0001
batch_size = 128

logits = model()
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=train_y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(train_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

dataset = readRecords("test.tfrecords", batch_size, epochs=1000)

# 指定使用显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.85  # 占用GPU90%的显存

with tf.Session(config=tf_config) as sess:
    next_batch = dataset.get_next()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    i = 0
    while True:
        batch_data = sess.run(next_batch)
        feed_dict = {train_x:batch_data["image"], train_y:batch_data["label"]}
        _, acc, loss = sess.run([train_step, accuracy, cross_entropy], feed_dict=feed_dict)
        print("epoch-%i   acc:%f,loss:%f" % (i,acc, loss))
        i += 1


