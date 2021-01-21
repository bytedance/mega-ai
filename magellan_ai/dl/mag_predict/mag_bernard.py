# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import grpc
import os


def show_func():
    print("+--------------------------+")
    print("|predict method            |")
    print("+--------------------------+")
    print("|1.tf_saved_model          |")
    print("|2.bernard_predict         |")
    print("+--------------------------+")


def tf_saved_model(x_train, x_valid, y_train, y_valid,
                   model_path, model_version=1):
    """save tensorflow model.

    Parameters
    ----------
    x_train : DataFrame
        Train feature set

    x_valid : str
        Validation feature set

    y_train : str
        Training label set

    y_valid :
        Validation label set

    model_path : string
        Model save path

    model_version : int
        The version information of the model

    Examples
    ---------
    >>> mag_bernard.tf_saved_model(x_train, x_valid,
    ... y_train, y_valid, model_path, model_version)
    """

    tf_clf = keras.Sequential()
    tf_clf.add(keras.layers.Dense(units=16,
                                  input_shape=(x_train.shape[1],),
                                  activation='relu'))

    tf_clf.add(keras.layers.Dense(1, activation='sigmoid'))

    tf_clf.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    tf_clf.fit(x=x_train.values,
               y=y_train.values,
               batch_size=1000,
               epochs=100,
               validation_data=(x_valid.values, y_valid.values),
               verbose=0)

    export_path = os.path.join(tf.compat.as_bytes(model_path),
                               tf.compat.as_bytes(str(model_version)))
    tf_clf.save(export_path)


def bernard_predict(x, hosts, input_name, output_name,
                    model_name, model_signature_name):
    """call bernard server model.

    Parameters
    ----------
    x : DataFrame
        Test feature set

    hosts : str
        Bernard server's hosts

    input_name : str
        input data name

    output_name : string
        output data name

    model_name : str
        model name

    model_signature_name: str
        model signature name

    num_tests: int
        Number of samples to be tested


    Returns
    --------
    res : DataFrame
        Model prediction results

    Examples
    ---------
    >>> mag_bernard.bernard_predict(x, hosts, input_name, output_name,
    ... model_name, model_signature_name)
    """

    test_images = x.values

    # connect and call rpc server
    channel = grpc.insecure_channel(hosts)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    req = predict_pb2.PredictRequest()
    req.model_spec.name = model_name
    req.model_spec.signature_name = model_signature_name

    pred_li = []
    for i in range(x.shape[0]):
        test_image = np.expand_dims(test_images[i], 0)
        cur_test_image = test_image/225.0
        cur_test_image = cur_test_image.astype(np.float32)
        req.inputs[input_name].CopyFrom(
            tf.make_tensor_proto(cur_test_image,
                                 shape=list(cur_test_image.shape)))

        res = stub.Predict(req, 3.0)
        pred_li.append(res.outputs[output_name].float_val[0])
        print("\rcalculation completed {:.2%}".format((i+1)/num_tests), end="")

    return pd.DataFrame(pred_li, columns=["prob"])
