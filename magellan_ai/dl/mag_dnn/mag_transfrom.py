# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals
import tensorflow as tf
import pandas as pd


def serialize_example(feat_data):
    """ 将tf.Tensor构成的tuple转成序列化tf.Data.Examples

    Parameters
    -----------
    feat_data: tuple
               由数据每行的各个特征值构成的元组

    Returns
    ---------
    example_proto.SerializeToString(): bytes
                                       将example序列化后的结果

    Examples
    ----------
    >>> serialize_example(feat_data)
    b'\n\xe9\x01\n\x12\n\tfeature_2\x12\x05\x1a\x03\n\x01-\n\x12\n\tfeature_6\x12\x05\x1a\x03\n\x01\r\n\x12\n\
    tfeature_7\x12\x05\x1a\x03\n\x01\x00\n\x12\n\tfeature_0\x12\x05\x1a\x03\n\x01\x01\n\x16\n\nfeature_10\x12\
    x08\x12\x06\n\x04\x00\x00\x00@\n\x12\n\tfeature_9\x12\x05\x1a\x03\n\x01\x00\n\x15\n\tfeature_1\x12\x08\x12\
    x06\n\x04\xe0 D?\n\x15\n\tfeature_4\x12\x08\x12\x06\n\x04=\x90M?\n\x15\n\tfeature_5\x12\x08\x12\x06\n\x04\
    x00\x80\x0eF\n\x12\n\tfeature_3\x12\x05\x1a\x03\n\x01\x02\n\x12\n\tfeature_8\x12\x05\x1a\x03\n\x01\x06'

    """


    feature_internal={}

    for index, feature in enumerate(feat_data):

        # 构建临时的特征名称
        feat_name = "feature_" + str(index)

        # 将特征封装成指定的三种类型，然后编码成特定的feature格式
        if feature.dtype in (tf.bool, tf.int32, tf.uint32, tf.int64, tf.uint64):
            feature_internal[feat_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))

        elif feature.dtype in (tf.float32, tf.float64):
            feature_internal[feat_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))

        elif feature.dtype in (tf.string, tf.byte):
            feature_internal[feat_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_internal))
    return example_proto.SerializeToString()


def tf_encode(input_path, filetype, output_path="", index_col=None):
    """将各种文件类型转化成TFRecord格式文件

    Parameters
    -----------
    input_path: str
                csv文件的输入路径

    filetype: str
              输出文件类型, 目前只提供csv转化成tfrecord

    output_path: str
                 tfrecord文件的输出路径，默认是csv文件的输入路径

    index_col: int, default=None
               选择数据的第几列作为索引


    Example
    ---------
    >>> input_path = "./xxx.csv"
    >>> filetype = "csv"
    >>> output_path = "./yyy.tfrecord"
    >>> index_col = 0
    >>> tf_encode(input_path, filetype, output_path, index_col)
    """

    # 如果导出路径为空，那么默认在输入路径下创建tf-record
    if len(output_path) == 0:
        # 将输入路径按照斜线分割
        input_li = input_path.split("/")

        # 将最后一层的文件名替换成.tfrecords
        csv_name = input_li.pop(-1)

        # 将csv_name的csv部分弹出
        name_li = csv_name.split(".")
        name_li.pop(-1)
        name = "".join(name_li)

        # 将csv_name的csv部分弹出
        tfrecords_name = "".join(name) + ".tfrecords"

        # 拼接到原来路径
        input_li.append(tfrecords_name)
        output_path = "/".join(input_li)

    # 根据文件类型指定读取方式
    if filetype == "csv":
        data_df = pd.read_csv(input_path, index_col=index_col)

    data_df.fillna(0, inplace=True)  # 之后需要判断处理缺失值

    # 用数组构成的元组，它返回元组的数据集
    data_tuple = tuple([data_df[col].values for col in data_df.columns])
    features_dataset = tf.data.Dataset.from_tensor_slices(data_tuple)

    # 将数据按照行进行处理
    def generator():
        for features in features_dataset:
            yield serialize_example(features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())

    writer = tf.data.experimental.TFRecordWriter(output_path)
    writer.write(serialized_features_dataset)


def tf_decode(input_path, feat_types, output_path="", feat_names=[]):
    """将各种文件类型转化成TFRecord格式文件

    Parameters
    -----------
    input_path: str
                tfrecord文件的输入路径

    feat_types: list
                特征类型列表

    output_path: str
                 csv文件的输出路径，默认是tfrecord文件的输入路径

    feat_names: list, default=[]
                特征名称列表


    Example
    ---------
    >>> input_path = "./xxx.csv"
    >>> filetype = "csv"
    >>> output_path = "./yyy.tfrecord"
    >>> index_col = 0
    >>> tf_encode(input_path, filetype, output_path)
    """

    feats_len = len(feat_types)

    # 如果导出路径为空，那么默认在输入路径下创建csv
    if len(output_path) == 0:
        # 将输入路径按照斜线分割
        input_li = input_path.split("/")

        # 将最后一层的文件名替换成.tfrecords
        csv_name = input_li.pop(-1)

        # 将csv_name的csv部分弹出
        name_li = csv_name.split(".")
        name_li.pop(-1)
        name = "".join(name_li)

        # 将csv_name的csv部分弹出
        tfrecords_name = "".join(name) + ".csv"

        # 拼接到原来路径
        input_li.append(tfrecords_name)
        output_path = "/".join(input_li)

    # 创建TFRecordDataset文件
    raw_dataset = tf.data.TFRecordDataset(input_path)

    # 如果特征名没有指定，那么就默认按照feature_xxx指定
    columns=[]
    for i in range(feats_len):
        columns.append("feature_" + str(i))

    if len(feat_types)==0:
        # 如果特征类型没有指定，那么就统一按照string来构建特征类型列表
        feat_types = [tf.string] * feats_len

    # 创建特征描述字典{特征名称:(特征大小，特征类型)}
    feature_description={}
    for column, feat_type in zip(columns, feat_types):
        feature_description[column] = tf.io.FixedLenFeature([1], feat_type)

    # 使用上述特征描述字典来解析example，将序列化的example解析成正常的字符串
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    # 使用map将函数应用于每一行
    parsed_dataset = raw_dataset.map(_parse_function)

    res_df = pd.DataFrame()
    for feat_dict in parsed_dataset:
        new_row =[]
        for i in range(len(feat_dict)):
            feat_name = "feature_" + str(i)
            new_row.append(feat_dict[feat_name].numpy()[0])
        res_df = pd.concat([res_df, pd.DataFrame([new_row])], axis=0)

    res_df.columns = columns
    res_df.to_csv(output_path, index=False)


if __name__ == "__main__":

    # # 保存TFrecord
    # input_path = "../../../../data/cs-training.csv"
    # output_path = "../../../../data/test.tfrecords"
    #
    # # 将csv文件编码成tf-record格式文件并保存, 并且将csv第一列作为索引
    # tf_encode(input_path, "csv", output_path, index_col=0)

    # 读取TFRecord文件
    input_path = "../../../../data/test.tfrecords"
    output_path = "../../../../data/ppppp.tfrecords"

    # 将tf-record格式文件解码成csv文件并保存
    feat_names = ["SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
                  "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
                  "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
                  "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]

    feat_types = [tf.int64, tf.float32, tf.int64, tf.int64, tf.float32, tf.float32, tf.int64,
                  tf.int64, tf.int64, tf.int64, tf.float32]

    tf_decode(input_path, feat_types, output_path, feat_names)
