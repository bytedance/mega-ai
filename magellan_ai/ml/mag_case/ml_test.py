# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import pandas as pd
from magellan_ai.ml.mag_util import mag_uap, mag_metrics, mag_calibrate
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    # # show doc string
    # print(help(mag_metrics))

    # # Function visualization
    # mag_metrics.show_func()
    # mag_uap.show_func()
    # mag_calibrate.show_func()

    # load data test
    data_df = pd.read_csv("../../../data/cs-training.csv", index_col=0)
    # data_df = pd.read_csv("../../../data/titanic/train.csv")

    # calculate IV and coverage
    print(mag_metrics.cal_iv(data_df, "SeriousDlqin2yrs", bin_method="decision_tree"))
    print(mag_metrics.cal_feature_coverage(data_df))
    data_df.fillna(0, inplace=True)

    # model training test
    X, y = data_df.iloc[:, 1:], data_df["SeriousDlqin2yrs"]
    lr = LogisticRegression(penalty="l2", max_iter=1000, random_state=99)
    X.fillna(0, inplace=True)
    lr.fit(X, y)
    y_proba = lr.predict_proba(X)[:, 1]
    test_df = pd.DataFrame({"label": y, "proba": y_proba})

    # model calibration test
    res = mag_calibrate.isotonic_calibrate(
        test_df, proba_name="proba", label_name="label",
        is_poly=True, bin_method="chi_square", bin_value_method="mean")
    # res = score_calibrate(test_df, proba_name="proba")
    # res = gaussian_calibrate(test_df, "proba")

    print(res)
