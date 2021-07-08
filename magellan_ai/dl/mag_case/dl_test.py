# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals

from sklearn.model_selection import train_test_split
from magellan_ai.dl.mag_predict import mag_online
from magellan_ai.dl.mag_predict import mag_bernard
from magellan_ai.ml.mag_util import mag_metrics
import pandas as pd


if __name__ == "__main__":

    # ----------------------------------------------------
    # export model feature list
    input_path = "../../../data/caijing_profile_clean.xlsx"
    output_path = "../../../data/mkt_catboost_hive_feats.txt"
    with open("../../../data/dou_feat_list.txt", "r") as f:
        feat_str = f.read()
    feat_list = feat_str.split("\n")
    mag_online.export_feats_map(input_path, feat_list, output_path, convert_type="v2_hive_feat")

    # ----------------------------------------------------
    # Bernard server test code
    # label_name = "is_bind_pay_success"
    # data_df = pd.read_csv("../../../data/bankpay_sample.csv")
    # data_df.fillna(0, inplace=True)
    # dou_cover = mag_metrics.cal_feature_coverage(data_df.iloc[:, 2:])
    # dou_feats = dou_cover[dou_cover["coverage"] > 0.01]["feature"].tolist()
    # X, y = data_df[dou_feats], data_df[label_name]
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)
    # model_path = "./python_pay_model/"
    # hosts = "10.150.128.73:9333"
    # input_name = "dense_input"
    # output_name = "dense_1"
    # model_name = "default"
    # model_version = 1
    # model_signature_name = "serving_default"
    # print(mag_bernard.show_func())

    # mag_bernard.tf_saved_model(X_train, X_valid, y_train, y_valid,
    #                            model_path, model_version)
    # mag_bernard.bernard_predict(X_train, hosts, input_name, output_name,
    #                             model_name, model_signature_name, 1000)

    # ----------------------------------------------------
    # deepFM module test code
    # dealing with numerical and categorical features
    # X_index, X_value = X.copy(deep=True), X.copy(deep=True)
    # feat_dict, feat_size = {}, 0
    # for dou_feat in dou_feats:
    #     num_unique = data_df[dou_feat].nunique()
    #     if num_unique < 10:
    #         unique_val = data_df[dou_feat].unique()
    #         feat_dict[dou_feat] = dict(zip(unique_val, range(
    #             feat_size, feat_size + len(unique_val))))
    #         feat_size += len(unique_val)
    #         X_index[dou_feat] = X_index[dou_feat].map(feat_dict[dou_feat])
    #         X_value[dou_feat] = 1
    #     else:
    #         feat_dict[dou_feat] = feat_size
    #         feat_size += 1
    #         X_index[dou_feat] = feat_dict[dou_feat]
    #
    # X_train_index, X_valid_index, X_train_value, X_valid_value, y_train, \
    #     y_valid = train_test_split(X_index, X_value, y, test_size=0.1)
    #
    # dmf = DeepFM(feature_size=feat_size, field_size=len(dou_feats), epoch=4,
    #              embedding_size=8, verbose=1, l2_reg=0.2, optimizer="adam")
    #
    # dmf.fit(x_train_index=X_train_index, x_train_value=X_train_value,
    #         y_train=y_train, x_valid_index=X_valid_index,
    #         x_valid_value=X_valid_value, y_valid=y_valid,
    #         refit=False, refit_epochs=50, refit_verbose=2)
    #
    # y_pred = dmf.predict(X_train_index,  X_train_value)

    # ----------------------------------------------------
    # TF_record test code
    # parquet/csv save TF_record
    # input_path = "../../../data/part.snappy.parquet"
    # input_path2 = "../../../data/test_tfrecord.csv"
    # output_path = "../../../data/ttt.tfrecord"
    # feat_path = "../../../data/ttt_featinfo.csv"
    # mag_transform.tf_encode(input_path, "parquet", feat_path, output_path)
    # mag_transform.tf_encode(input_path2, "csv", feat_path, output_path)

    # read saved TFRecord file
    # input_path = "../../../data/ttt.tfrecord"
    # output_path = "../../../data/ccc.csv"
    # feat_info_path = "../../../data/ttt_featinfo.csv"
    # mag_transform.tf_decode(input_path, feat_info_path, "csv", output_path)
