# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import pandas as pd
from magellan_ai.ml.mag_util import mag_uap, mag_metrics, mag_calibrate
from sklearn.linear_model import LogisticRegression


# 计算auc
def case_metrics_auc(path, prob_col, label_col):
    df = pd.read_csv(path)
    auc = mag_metrics.cal_auc(df.loc[:, prob_col], df.loc[:, label_col])
    print(auc)


if __name__ == "__main__":

    print(help(mag_metrics))

    # 功能可视化
    mag_metrics.show_func()
    mag_uap.show_func()
    mag_calibrate.show_func()

    # 导入数据
    data_df = pd.read_csv("/Users/bytedance/ByteCode"
                          "/magellan_ai/data/cs-training.csv", index_col=0)

    # 计算iv值和覆盖率
    print(mag_metrics.cal_iv(data_df, "SeriousDlqin2yrs"))
    print(mag_metrics.cal_feature_coverage(data_df))

    # 模型训练
    X = data_df.iloc[:, 1:]
    y = data_df["SeriousDlqin2yrs"]
    lr = LogisticRegression(penalty="l2", random_state=99)
    # X.fillna(0, inplace=True)
    # lr.fit(X, y)
    # y_proba = lr.predict_proba(X)[:, 1]
    # test_df = pd.DataFrame({"label": y, "proba": y_proba})
    #
    # # 测试模型校准模块
    # res = mag_calibrate.isotonic_calibrate(
    #     test_df, proba_name="proba", label_name="label",
    #     is_poly=True, bin_method="chi_square", bin_value_method="mean")
    # # res = score_calibrate(test_df, proba_name="proba")
    # # res = gaussian_calibrate(test_df, "proba")
    #
    # print(res)
