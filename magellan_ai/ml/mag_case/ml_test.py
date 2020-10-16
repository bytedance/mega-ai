# coding: utf-8
from __future__ import absolute_import, division, \
    print_function, unicode_literals
import pandas as pd
from magellan_ai.ml.mag_util import mag_metrics
from magellan_ai.ml.mag_util import mag_modelonline
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    # # show doc string
    # print(help(mag_metrics))

    # # Function visualization
    # mag_metrics.show_func()
    # mag_uap.show_func()
    # mag_calibrate.show_func()

    # load data test
    # data_df = pd.read_csv("../../../data/cs-training.csv", index_col=0)
    data_df = pd.read_csv("../../../data/titanic/train.csv")

    # calculate IV and coverage
    print(mag_metrics.cal_iv(data_df, "Survived", bin_method="same_frequency"))
    print(mag_metrics.cal_feature_coverage(data_df))
    data_df.fillna(0, inplace=True)

    # model training test
    X, y = data_df.iloc[:, 1:], data_df["Survived"]
    lr = LogisticRegression(penalty="l2", max_iter=1000, random_state=99)
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
    #
    # print(res)

    # feat_path = "./featinfo.json"
    # feat_names = ["aaa", "bbb", "ccc"]
    # model_name = "hhh.bin"
    # dump_feats_json(model_name, feat_names, feat_path)

    # input_path = "../../../data/财经用户画像数据需求文档.xlsx"
    # output_path = "../../../data/caijing_profile_clean.xlsx"
    # clean_table(input_path, output_path)

    input_path = "../../../data/caijing_profile_clean.xlsx"
    output_path = "../../../data/schema_infos.txt"
    feat_names = ['account_stability_score_min', 'device_days_max',
                  'device_days_avg', 'device_count_avg',
                  'device_days_min', 'device_count_min',
                  'account_stability_score_max', 'device_count_max',
                  'account_stability_score_avg']
    wrong_sheet = ["接入AppId", "反欺诈模型V2-莫菲雨",
                   "反欺诈策略-张馨予-20200617", "交易模型V1-莫菲雨"]
    mag_modelonline.get_feats_map(input_path, feat_names,
                                  wrong_sheet, output_path)
