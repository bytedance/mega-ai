# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals

from sklearn.model_selection import train_test_split
from magellan_ai.dl.mag_dnn.mag_mining import DeepFM
from magellan_ai.dl.mag_dnn import mag_transform
from magellan_ai.ml.mag_util import mag_metrics
from sklearn.metrics import roc_auc_score
import pandas as pd


if __name__ == "__main__":

    # load deepFM data
    label_name = "is_bind_pay_success"
    data_df = pd.read_csv("../../../data/bankpay_sample.csv")
    data_df.fillna(0, inplace=True)
    dou_cover = mag_metrics.cal_feature_coverage(data_df.iloc[:, 2:])
    dou_feats = dou_cover[dou_cover["coverage"] > 0.01]["feature"].tolist()
    X, y = data_df[dou_feats], data_df[label_name]
    X_index, X_value = X.copy(deep=True), X.copy(deep=True)

    # dealing with numerical and categorical features
    feat_dict, feat_size = {}, 0
    for dou_feat in dou_feats:
        num_unique = data_df[dou_feat].nunique()
        if num_unique < 10:
            unique_val = data_df[dou_feat].unique()
            feat_dict[dou_feat] = dict(zip(unique_val, range(feat_size, feat_size + len(unique_val))))
            feat_size += len(unique_val)
            X_index[dou_feat] = X_index[dou_feat].map(feat_dict[dou_feat])
            X_value[dou_feat] = 1
        else:
            feat_dict[dou_feat] = feat_size
            feat_size += 1
            X_index[dou_feat] = feat_dict[dou_feat]

    X_train_index, X_valid_index, X_train_value, X_valid_value, y_train, y_valid = train_test_split(
        X_index, X_value, y, test_size=0.1)

    dmf = DeepFM(feature_size=feat_size, field_size=len(dou_feats), epoch=4,
                 embedding_size=8, verbose=1)
    dmf.fit(x_train_index=X_train_index, x_train_value=X_train_value, y_train=y_train,
            x_valid_index=X_valid_index, x_valid_value=X_valid_value, y_valid=y_valid,
            refit=False, refit_epochs=50, refit_verbose=2)
    y_pred = dmf.predict(X_train_index,  X_train_value)
    res = roc_auc_score(y_train, y_pred)
    print("model predict roc:", res)
    weight_dict = {weight_name: dmf.sess.run(weight_value) for weight_name, weight_value in dmf.weights.items()}

    # # parquet/csv 保存TFrecord
    # input_path = "../../../data/part.snappy.parquet"
    # # input_path = "../../../data/test_tfrecord.csv"
    # output_path = "../../../data/ttt.tfrecord"
    # feat_path = "../../../data/ttt_featinfo.csv"
    # mag_transform.tf_encode(input_path, "parquet", feat_path, output_path)

    # 读取保存的TFRecord文件
    # input_path = "../../../data/ttt.tfrecord"
    # output_path = "../../../data/ccc.csv"
    # feat_info_path = "../../../data/ttt_featinfo.csv"
    #
    # # 将tf-record格式编码成csv文件并保存, 并且将csv第一列作为索引
    # mag_transform.tf_decode(input_path, feat_info_path, "csv", output_path)
    #
    # df = pd.read_parquet("../../../data/ccc.parquet")
    # print(df.shape)
    # print(df.head(5))
