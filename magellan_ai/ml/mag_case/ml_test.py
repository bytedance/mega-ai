# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals
from magellan_ai.ml.mag_util import mag_metrics, mag_online
from magellan_ai.ml.mag_util.mag_mining import DeepFM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

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

    # show doc string
    # print(help(mag_metrics))

    # # Function visualization
    # mag_metrics.show_func()
    # mag_uap.show_func()
    # mag_calibrate.show_func()

    # load data test
    # data_df = pd.read_csv("../../../data/cs-training.csv", index_col=0)
    # data_df = pd.read_csv("../../../data/titanic/train.csv")
    #
    # # calculate IV and coverage
    # print(mag_metrics.cal_iv(data_df, "Survived", bin_method="same_frequency"))
    # print(mag_metrics.cal_feature_coverage(data_df))
    # data_df.fillna(0, inplace=True)

    # # model training test
    # X, y = data_df.iloc[:, 1:], data_df["Survived"]
    # lr = LogisticRegression(penalty="l2", max_iter=1000, random_state=99)
    # X.fillna(0, inplace=True)
    # lr.fit(X, y)
    # y_proba = lr.predict_proba(X)[:, 1]
    # test_df = pd.DataFrame({"label": y, "proba": y_proba})
    #
    # # model calibration test
    # res = mag_calibrate.isotonic_calibrate(
    #     test_df, proba_name="proba", label_name="label",
    #     is_poly=True, bin_method="chi_square", bin_value_method="mean")
    # res = score_calibrate(test_df, proba_name="proba")
    # res = gaussian_calibrate(test_df, "proba")

    # mag_online test case
    # input_path = "../../../data/caijing_profile_clean.xlsx"
    # origin_path = "../../../data/财经用户画像数据需求文档.xlsx"
    # feat_path = "../../../data/feat_json_format.json"
    # output_path = "../../../data/make_json.txt"
    # model_name = "templete_model.bin"
    #
    # with open("../../../data/feats_list.txt") as f:
    #     feats_list = f.read().split("\n")

    # mag_online.clean_table(origin_path, input_path)
    # mag_online.dump_feats_json(model_name, feats_list, feat_path)
    # mag_online.export_feats_map(input_path, feats_list, output_path, convert_type="v1_map_get")
    # mag_online.export_feats_map(input_path, feats_list, output_path, convert_type="v2_feat_group")
    # mag_online.export_feats_map(input_path, feats_list, output_path, convert_type="v2_make_json")
    # mag_online.export_feats_map(input_path, feats_list, output_path, convert_type="v3_json")
