# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals

from magellan_ai.dl.mag_dnn import mag_transfrom


if __name__ == "__main__":

    # # 保存TFrecord
    # input_path = "../../../../data/test.csv"
    # output_path = "../../../../data/test.tfrecords"
    # feat_path = "../../../../data/test_featinfo.tfrecords"
    # # data_df = pd.read_csv(input_path)
    # # print(data_df)
    # # data_df.fillna(0, inplace=True)

    # # 将csv文件编码成tf-record格式文件并保存, 并且将csv第一列作为索引
    # mag_transfrom.tf_encode(input_path, "csv", output_path, index_col=0)

    # # 读取保存的TFRecord文件
    input_path = "../../../../data/test.tfrecords"
    output_path = "../../../../data/ccccccc.csv"
    feat_info_path = "../../../../data/test_featinfo.csv"

    # 将tf-record格式编码成csv文件并保存, 并且将csv第一列作为索引
    mag_transfrom.tf_decode(input_path, feat_info_path, output_path)

    # data_df = pd.read_csv(input_path)
    # print(data_df)
    # data_df.fillna(0, inplace=True)
