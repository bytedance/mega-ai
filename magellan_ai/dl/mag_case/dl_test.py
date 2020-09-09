# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals

from magellan_ai.dl.mag_dnn import mag_transform
import pandas as pd


if __name__ == "__main__":

    # # parquet/csv 保存TFrecord
    # input_path = "../../../data/part.snappy.parquet"
    # # input_path = "../../../data/test_tfrecord.csv"
    # output_path = "../../../data/ttt.tfrecord"
    # feat_path = "../../../data/ttt_featinfo.csv"
    # mag_transform.tf_encode(input_path, "parquet", feat_path, output_path)

    # 读取保存的TFRecord文件
    input_path = "../../../data/ttt.tfrecord"
    output_path = "../../../data/ccc.csv"
    feat_info_path = "../../../data/ttt_featinfo.csv"

    # 将tf-record格式编码成csv文件并保存, 并且将csv第一列作为索引
    mag_transform.tf_decode(input_path, feat_info_path, "csv", output_path)

    # df = pd.read_parquet("../../../data/ccc.parquet")
    # print(df.shape)
    # print(df.head(5))
