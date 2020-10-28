# -*- coding: utf-8 -*-


from tensorflow_serving.apis import prediction_service_pb2_grpc  # 用来处理grpc数据的脚本
from tensorflow_serving.apis import predict_pb2  # 用来处理protobuf格式数据的脚本
from tensorflow import keras
import tensorflow as tf
import numpy as np
import grpc
import sys


"""
    Author: huangning
    Date: 2020/10/12
    Function: 用Python调用Bernard服务
"""

# 新增命令行参数
tf.compat.v1.app.flags.DEFINE_string("input_data_path", "/Users/huangning/.keras/datasets/fashion-mnist", "测试样本路径")
tf.compat.v1.app.flags.DEFINE_integer("concurrency", 1, "并发请求的最大数量")
tf.compat.v1.app.flags.DEFINE_integer("num_tests", 1000, "要测试的样本量")
tf.compat.v1.app.flags.DEFINE_string("server", "10.30.38.95:9200", "服务的ip和端口号")
FLAGS = tf.compat.v1.app.flags.FLAGS


def cal_accuracy(hostport, input_path, concurrency, num_tests):

    # 连接 rpc 服务器
    channel = grpc.insecure_channel(hostport)

    # 调用 rpc 的服务(返回预测服务的对象)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 导入测试集
    # test_images, test_labels = keras.datasets.fashion_mnist.load_data()[1]
    test_images, test_labels = keras.datasets.mnist.load_data()[1]

    # 建立请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    # request.model_spec.signature_name = 'serving_default'
    request.model_spec.signature_name = 'predict_images'

    # 统计预测值和真实值相等的个数
    same_count = 0
    for i in range(num_tests):
        test_image = np.expand_dims(test_images[i], 0)
        cur_test_image = test_image/225.0
        cur_test_image = cur_test_image.astype(np.float32)
        request.inputs["images"].CopyFrom(
            tf.make_tensor_proto(cur_test_image, shape=list(cur_test_image.shape)))

        # 发送预测请求，将10秒设置为超时请求
        result = stub.Predict(request, 5.0)
        pred, test_label = np.argmax(result.outputs["softmax"].float_val), test_labels[i]
        same_count = same_count + 1 if pred == test_label else same_count
        print("\r已完成{:.2%}".format((i+1)/num_tests), end="")

    # 计算准确率
    return same_count / num_tests


def main(_):

    # 参数列表异常值判断
    if FLAGS.num_tests > 1000:
        print("要测试的样本量应小于1k")
        sys.exit(-1)
    if not FLAGS.server:
        print("请指定服务的IP和端口号")
        sys.exit(-1)

    # 计算准确率
    accuracy_rate = cal_accuracy(FLAGS.server, FLAGS.input_data_path, FLAGS.concurrency, FLAGS.num_tests)
    print('\n准确率的计算结果为:{:.2%}'.format(accuracy_rate))


if __name__ == "__main__":

    print("测试连接Bernard服务")
    tf.compat.v1.app.run()
