# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np


def show_func():
    print("+-------------------------------+")
    print("|feature evaluation methods     |")
    print("+-------------------------------+")
    print("|1.tf_encode                    |")
    print("|2.tf_decode                    |")
    print("+-------------------------------+")


def serialize_example(feat_data):
    """ Let the tuple of tf.Tensor convert into tf.Data.Examples of serillization

    Parameters
    -----------
    feat_data: tuple
        A tuple consisting of the feature values of each row of data

    Returns
    ---------
    example_proto.SerializeToString(): bytes
        The result of serializing tf.Data.Examples

    Examples
    ----------
    >>> serialize_example(feat_data)
    b'\n\xe9\x01\n\x12\n\tfeature_2\x12\x05\x1a\x03\n\x01-\n\x12\n\tfeature_6\x12
    \x05\x1a\x03\n\x01\r\n\x12\n\tfeature_7\x12\x05\x1a\x03\n\x01\x00\n\x12\n\
    tfeature_0\x12\x05\x1a\x03\n\x01\x01\n\x16\n\nfeature_10\x12\x08\x12\x06\n\
    x04\x00\x00\x00@\n\x12\n\tfeature_9\x12\x05\x1a\x03\n\x01\x00\n\x15\n\
    tfeature_1\x12\x08\x12\x06\n\x04\xe0 D?\n\x15\n\tfeature_4\x12\x08\
    x12\x06\n\x04=\x90M?\n\x15\n\tfeature_5\x12\x08\x12\x06\n\x04\x00\
    x80\x0eF\n\x12\n\tfeature_3\x12\x05\x1a\x03\n\x01\x02\n\x12\n\
    tfeature_8\x12\x05\x1a\x03\n\x01\x06'

    """

    feature_internal = {}

    for index, feature in enumerate(feat_data):

        feat_name = "feature_" + str(index)
        if feature.dtype in (tf.bool, tf.int32, tf.uint32,
                             tf.int64, tf.uint64):
            feature_internal[feat_name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[feature]))
        elif feature.dtype in (tf.float32, tf.float64):
            feature_internal[feat_name] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[feature]))
        elif feature.dtype == tf.string:

            # Let eagerTensor convert into bytes
            if isinstance(feature, type(tf.constant(0))):
                feature = feature.numpy()
            feature_internal[feat_name] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[feature]))

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature_internal))
    return example_proto.SerializeToString()


def tf_encode(input_path, filetype, feat_info_path="",
              output_path="", index_col=None):
    """Convert various format files into tfrecords format files

    Parameters
    -----------
    input_path: str
        The input path of the file.

    filetype: {"csv", "parquet"}
        The input file type.

    feat_info_path: str, default=""
        The output path of feature information. he default is the path of the read in file.

    output_path: str, default=""
        The output path of the tfrecord file. The default is the path of the read in file.

    index_col: int, default=None
        Which column of data is selected as the index.

    Returns
    ---------
    output_file: tfrecord
        Tfrecord format file

    feat_info_file : csv
        Feature information file

    Example
    ---------
    >>> input_path = "/path/to/x1.csv"
    >>> output_path = "/path/to/y1.tfrecord"
    >>> feat_path = "/path/to/x1_featinfo.csv"
    >>> mag_transform.tf_encode(input_path, "csv", output_path, feat_path, 0)
    >>> input_path2 = "/path/to/x2.parquet"
    >>> output_path2 = "/path/to/y2.tfrecord"
    >>> feat_path2 = "/path/to/x2_featinfo.csv"
    >>> mag_transform.tf_encode(input_path2, "parquet", output_path, feat_path)
    """

    if len(output_path) == 0:
        input_li = input_path.split("/")
        csv_name = input_li.pop(-1)
        name_li = csv_name.split(".")
        name_li.pop(-1)
        tfrecords_name = "".join(name_li) + ".tfrecord"
        input_li.append(tfrecords_name)
        output_path = "/".join(input_li)

    if len(feat_info_path) == 0:
        input_li = input_path.split("/")
        feat_temp = input_li.pop(-1)
        feat_li = feat_temp.split(".")
        feat_li.pop(-1)
        feat_filename = "".join(feat_li) + "_featinfo.csv"
        input_li.append(feat_filename)
        feat_path = "/".join(input_li)

    # Specifies the read method according to the file type
    if filetype == "csv":
        data_df = pd.read_csv(input_path,
                              index_col=index_col)
    elif filetype == "parquet":
        data_df = pd.read_parquet(input_path,
                                  engine="pyarrow")
        data_df.fillna("null", inplace=True)
    else:
        raise Exception("The current file format does not support tfrecords conversion")

    # The tuple is composed of array to ensure that the later slice still retains the data format
    data_tuple = tuple([data_df[col].values for col in data_df.columns])

    # Cut the first dimension of array
    features_dataset = tf.data.Dataset.from_tensor_slices(data_tuple)

    # Save the feature type and feature name for decode
    feat_info = []
    for features in features_dataset:
        for col_name, feat in zip(data_df.columns, features):
            feat_info.append([col_name, feat.dtype])
        break
    feat_info_df = pd.DataFrame(feat_info, columns=["feat_name", "feat_type"])
    feat_info_df.to_csv(feat_info_path, index=False)

    # Process data as rows
    def generator():
        for features in features_dataset:
            yield serialize_example(features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())

    writer = tf.data.experimental.TFRecordWriter(output_path)
    writer.write(serialized_features_dataset)


def tf_decode(input_path, feat_info_path, output_type, output_path=""):
    """Convert tfrecord format file to specified format file.

    Parameters
    -----------
    input_path: str
        The path to the tfrecord file

    feat_info_path: str
        The Path of feature information

    output_type: {"csv", "parquet"}
        The file type of the output file

    output_path: str
        The path to save the output file. The default is the path to read in the file

    Returns
    ---------
    output_file: {"csv", "parquet"}
        File of the specified type

    Example
    ---------
    >>> input_path = "/path/to/x1.tfrecord"
    >>> feat_info_path = "/path/to/x1_featinfo.csv"
    >>> output_path = "/path/to/y1.csv"
    >>> mag_transfrom.tf_decode(input_path, feat_info_path, output_path, "csv")
    >>> input_path2 = "/path/to/x2.tfrecord"
    >>> feat_info_path2 = "/path/to/x2_featinfo.csv"
    >>> output_path2 = "/path/to/y2.parquet"
    >>> mag_transfrom.tf_decode(input_path, feat_info_path, output_path, "parquet")
    """

    feat_df = pd.read_csv(feat_info_path)
    feat_names, feat_nptypes = \
        feat_df["feat_name"].values, feat_df["feat_type"].values

    # Convert the saved feature information into numpy object
    feat_types = []
    for feat in feat_nptypes:
        feat_type = feat.split("'")[1]
        if feat_type == "string":
            feat_type = "object"
        if feat_type == "float64":
            feat_type = "float32"
        feat_types.append(np.dtype(feat_type))

    feats_len = len(feat_types)

    # If the export path is empty, the specified format file is created in the input path by default
    if len(output_path) == 0:
        input_li = input_path.split("/")
        csv_name = input_li.pop(-1)
        name_li = csv_name.split(".")
        name_li.pop(-1)
        if output_type == "csv":
            tfrecords_name = "".join(name_li) + ".csv"
        elif output_type == "parquet":
            tfrecords_name = "".join(name_li) + ".parquet"
        else:
            raise Exception("current type of file is not supported")

        input_li.append(tfrecords_name)
        output_path = "/".join(input_li)

    # Create TFRecordDataset object
    raw_dataset = tf.data.TFRecordDataset(input_path)

    columns = []
    for i in range(feats_len):
        columns.append("feature_" + str(i))

    # If the feature type is not specified, the list of feature types is constructed according to string
    if len(feat_types) == 0:
        feat_types = [tf.string] * feats_len

    # Create feature description dictionary {feature Name: (feature size, feature type)}
    feature_description = {}
    for column, feat_type in zip(columns, feat_types):
        feature_description[column] = tf.io.FixedLenFeature([1], feat_type)

    # Use the above feature description dictionary to parse the
    # tf.Data.Examples and parse the serialized tf.Data.Examples into a normal string
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    total_row = []
    for index, feat_dict in enumerate(parsed_dataset):
        new_row = []
        for i in range(len(feat_dict)):
            feat_name = "feature_" + str(i)
            cur_feat_value = feat_dict[feat_name].numpy()[0]
            if type(cur_feat_value) == np.dtype("bytes"):
                cur_feat_value = str(cur_feat_value, encoding='utf-8')
            new_row.append(cur_feat_value)
        total_row.append(new_row)
    res_df = pd.DataFrame(total_row, columns=feat_names)

    # If the feature name is not specified, the default is `feature_xxx` designated
    res_df.columns = feat_names

    if output_type == "csv":
        res_df.to_csv(output_path, index=False)
    elif output_type == "parquet":
        res_df.to_parquet(output_path, index=False)
    else:
        raise Exception("The current file format does not support saving")
