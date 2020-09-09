# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals

from magellan_ai.dl.mag_dnn import mag_transform
import pandas as pd


if __name__ == "__main__":

    # # 保存TFrecord
    # input_path = "../../../data/part.snappy.parquet"
    # output_path = "../../../../data/test.tfrecords"
    # data_df = pd.read_parquet(input_path, engine="pyarrow")
    # data_df.to_parquet("../../../data/xxx.parquet")
    # a = pd.read_parquet("../../../data/xxx.parquet")

    # # parquet保存TFrecord
    input_path = "../../../data/part.snappy.parquet"
    # input_path2 = "../../../data/cs-training.csv"
    output_path = "../../../data/ttt.tfrecords"
    feat_path = "../../../data/ttt_featinfo.tfrecords"
    mag_transform.tf_encode(input_path, "parquet", output_path, feat_path)

    # # 将csv文件编码成tf-record格式文件并保存, 并且将csv第一列作为索引
    # mag_transfrom.tf_encode(input_path, "csv", output_path, index_col=0)

    # # 读取保存的TFRecord文件
    # input_path = "../../../../data/test.tfrecords"
    # output_path = "../../../../data/ccccccc.csv"
    # feat_info_path = "../../../../data/test_featinfo.csv"

    # # 将tf-record格式编码成csv文件并保存, 并且将csv第一列作为索引
    # mag_transform.tf_decode(input_path, feat_info_path, output_path)
