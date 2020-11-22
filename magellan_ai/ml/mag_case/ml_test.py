# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals
from magellan_ai.ml.mag_util import mag_metrics
from magellan_ai.dl.mag_dnn.mag_mining import DeepFM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # show doc string
    print(help(mag_metrics))

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
