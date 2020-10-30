# -*- coding: utf-8 -*-

from __future__ import print_function
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential, Model


import tensorflow as tf
import numpy as np
import sys
import os

"""
    Author: huangning
    Date: 2020/10/08
    Target: 训练并且导出fashion_mnist图像多分类模型
"""

# flags中新增参数脚本
tf.compat.v1.app.flags.DEFINE_integer('training_iteration', 5, '训练迭代的次数.')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 100, '每批训练样本量大小.')
tf.compat.v1.app.flags.DEFINE_integer('model_version', 1, '模型版本号.')

# 目前该路径没有用上！！！
tf.compat.v1.app.flags.DEFINE_string('input_path', '/tmp', '数据存储路径.')
FLAGS = tf.compat.v1.app.flags.FLAGS

# 关闭eager模式
tf.compat.v1.disable_eager_execution()


def main(_):

    print("系统获取的参数列表为", sys.argv)

    # 参数列表异常判断
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('请使用正确的命令行参数语法: python3 mnist_saved_model.py '
              '[--training_iteration=x] [--model_version=y] model_export_dir')
        sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print('请将训练迭代次数.')
        sys.exit(-1)
    if FLAGS.batch_size <= 0:
        print("请将每批次训练样本指定为正数")
    if FLAGS.model_version <= 0:
        print('请将版本信息指定为正数.')
        sys.exit(-1)

    # <----------获取数据------------->
    # load_data()会自动将数据下载到 ~/.keras/datasets/fashion-mnist
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # <----------构建模型--------------->
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # <----------模型训练--------------->
    # 将像素值缩小至[0,1]，然后将其馈送到神经网络模型。
    train_images, test_images = train_images / 255.0, test_images / 255.0
    x_valid, x_train = train_images[:10000], train_images[10000:]
    y_valid, y_train = train_labels[:10000], train_labels[10000:]

    model.fit(x_train, y_train, epochs=FLAGS.training_iteration, batch_size=FLAGS.batch_size,
              validation_data=(x_valid, y_valid), verbose=1)

    # 结尾增加softmax层, 使得输出为为概率
    prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # 最后一个参数就是模型保存路径
    export_path_base = sys.argv[-1]

    # 构建模型导出路径并保存在export_path
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))

    print("查看模型的输入信息")
    print(prob_model.input)
    print("查看输出信息")
    print(prob_model.output)

    # 模型保存
    prob_model.save(export_path)


if __name__ == "__main__":

    # run方法找当前模块下的main函数，并且传入一个参数,
    # 如果把main中的下划线去掉，就表示这个main函数是不带参数的，所以会报错
    tf.compat.v1.app.run()
