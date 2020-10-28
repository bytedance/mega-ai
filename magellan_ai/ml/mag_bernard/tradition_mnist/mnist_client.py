"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import time
import grpc
import numpy
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import mnist_input_data

# from megaspark.bernard.tradition_mnist import mnist_input_data

# 新增命令行参数
tf.compat.v1.app.flags.DEFINE_string("input_data_path", "/tmp", "测试样本路径")
tf.compat.v1.app.flags.DEFINE_integer("concurrency", 1, "并发请求的最大数量")
tf.compat.v1.app.flags.DEFINE_integer("num_tests", 1000, "要测试的样本量")
tf.compat.v1.app.flags.DEFINE_string("server", "10.30.38.95:9200", "服务的ip和端口号")
FLAGS = tf.compat.v1.app.flags.FLAGS


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """

    def _callback(result_future):
        """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('..')
            sys.stdout.flush()
            response = numpy.array(
                result_future.result().outputs['scores'].float_val)
            prediction = numpy.argmax(response)
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()

    return _callback


def do_inference(hostport, work_dir, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """


    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)

    for _ in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'default'
        request.model_spec.signature_name = 'predict_images'

        image, label = test_data_set.next_batch(1)

        request.inputs['images'].CopyFrom(
            tf.make_tensor_proto(image[0], shape=[1, image[0].size]))
        # result_counter.throttle()

        # 发出预测请求
        result_future = stub.Predict.future(request, 1.0)  # 5 seconds
        result = stub.Predict(request, 5)
        print(result)

        return 0


def cal_accuracy(hostport, input_path, concurrency, num_tests):

    # 连接 rpc 服务器
    channel = grpc.insecure_channel(hostport)

    # 调用 rpc 的服务(返回预测服务的对象)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 导入测试集
    test_data_set = mnist_input_data.read_data_sets(input_path).test

    # 建立请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'predict_images'

    # 统计预测值和真实值相等的个数
    same_count = 0
    for i in range(num_tests):

        image, label = test_data_set.next_batch(1)

        request.inputs['images'].CopyFrom(
          tf.make_tensor_proto(image[0], shape=[1, image[0].size]))

        # 发出预测请求
        result = stub.Predict(request, 5)
        pred, test_label = np.argmax(result.outputs["scores"].float_val), label
        same_count = same_count + 1 if pred == test_label else same_count
        print("\r计算已完成{:.2%}".format((i + 1) / num_tests), end="")

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


if __name__ == '__main__':
    tf.compat.v1.app.run()
