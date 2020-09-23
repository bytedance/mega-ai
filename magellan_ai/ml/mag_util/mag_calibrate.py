# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import six.moves as sm

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from magellan_ai.ml.mag_util.mag_metrics import chisquare_binning_boundary, \
    decisiontree_binning_boundary


def show_func():

    print("+--------------------------+")
    print("|calibrate method          |")
    print("+--------------------------+")
    print("|1.isotonic_calibrate      |")
    print("|2.gaussian_calibrate      |")
    print("|3.score_calibrate         |")
    print("+--------------------------+")


def isotonic_calibrate(input_df, proba_name, label_name,
                       is_poly=False, fit_num=3, bin_num=1000,
                       bin_method="same_frenquency", bin_value_method="mean"):
    """Isotonic regression calibration.

    Parameters
    ----------
    input_df : DataFrame
        There are two columns of data frame,
        one is prediction probability, the other is label value.

    proba_name : str
        Prediction probability name.

    label_name : str
        Label name.

    is_poly : bool
        Whether polynomial fitting is used for
        comparision with isotonic regression calibration results.

    fit_num : int
        The number of polynomial fitting.

    bin_num : int
        Maximum number of bins.

    bin_method : {'same_frequency','decision_tree','chi_square'}, \
                 default='same_frequency'
        bin value calculation method.

    bin_value_method : {'mean','medium','mode'}, default='mean'
        At present, there are average, median and mode.

    Returns
    --------
    input_df : DataFrame
        The calculation results of isotonic
        regression calibration (polynomial fitting results) are added.

    Examples
    ---------
    >>> df
            label     proba
    1           1  0.241217
    2           0  0.096250
    3           0  0.140861
    4           0  0.119471
    5           0  0.005491
    ...       ...       ...
    149996      0  0.019120
    149997      0  0.097704
    149998      0  0.020250
    149999      0  0.111851
    150000      0  0.024276
    >>> res = isotonic_calibrate(df, proba_name="proba",
    ... label_name="label", is_poly=True,  bin_method="same_frequency",
    ... bin_value_method="mean")
    >>> res
            label     proba  iso_pred  1_polynomial  2_polynomial  3_polynomial
    1           1  0.241217  0.222568      0.221545      0.193340      0.128076
    2           0  0.096250  0.088297      0.087469      0.079517      0.089461
    3           0  0.140861  0.119745      0.118871      0.105503      0.107274
    4           0  0.119471  0.103582      0.102732      0.092096      0.098910
    5           0  0.005491  0.007915      0.007204      0.014969      0.011603
    ...       ...       ...       ...           ...           ...           ...
    149996      0  0.019120  0.027562      0.026823      0.030497      0.035378
    149997      0  0.097704  0.089253      0.088424      0.080301      0.090098
    149998      0  0.020250  0.028784      0.028042      0.031468      0.036748
    149999      0  0.111851  0.098566      0.097724      0.087958      0.095979
    150000      0  0.024276  0.032668      0.031921      0.034559      0.041024
    """

    if bin_method == "same_frequency":

        # If the number of feature enumerations is less than k_part,
        # they are grouped directly according to the enumeration value
        if len(input_df[proba_name].unique()) < bin_num:
            boundary_list = [input_df[proba_name].min() -
                             0.001] + [input_df[proba_name].unique().sort()]
        else:
            cur_feat_interval = sorted(
                pd.qcut(input_df[proba_name], bin_num,
                        precision=3, duplicates="drop").unique())

            boundary_list = [cur_feat_interval[0].left] + \
                            [value.right for value in cur_feat_interval]

    elif bin_method == "decision_tree":
        boundary_list = decisiontree_binning_boundary(
            input_df, proba_name, label_name, bin_num)

    elif bin_method == "chi_square":

        # If the number of values of the feature is greater than 100,
        # it is necessary to discretize the feature into 100 values
        # at the same frequency to speed up the operation
        if len(input_df[proba_name].unique()) >= 100:
            cur_feat_interval = \
                pd.qcut(input_df[proba_name], 100, duplicates="drop")
            input_df[proba_name] = cur_feat_interval

            # According to the average of the left and right endpoints
            # of the partition interval as the discrete enumeration value,
            # the continuous feature is transformed into the discrete feature
            input_df[proba_name] = input_df[proba_name].apply(
                lambda x: float((x.left + x.right) / 2))

        boundary_list = chisquare_binning_boundary(
            input_df, proba_name, label_name, bin_num)
        input_df[proba_name] = input_df[proba_name].astype("float64")

    else:
        raise Exception("The current {} method is not "
                        "implemented".format(bin_method))

    # Store 2 dimensions coordinate points in the list. The first
    # dimension is the bin value and the second
    # dimension is the real passing rate
    coordinate_li = []
    for i in sm.range(len(boundary_list) - 1):

        group_df = input_df[(input_df[proba_name] > boundary_list[i])
                            & (input_df[proba_name] <= boundary_list[i + 1])]

        # Calculate the proportion of positive cases in the current group
        pos_ratio = sum(group_df[label_name] == 1) / group_df.shape[0] \
            if group_df.shape[0] != 0 else 1

        if group_df.shape[0] == 0:

            # If there is no element in the current group,
            # the right endpoint is selected by defaul
            temp = boundary_list[i + 1]
        elif bin_value_method == "mean":
            temp = group_df[proba_name].mean()
        elif bin_value_method == "medium":
            temp = group_df[proba_name].median()
        elif bin_value_method == "mode":

            # There may be multiple modes,
            # and the first one is selected by default
            temp = group_df[proba_name].mode()[0]
        else:
            raise Exception("bin_value_method entered "
                            "is not within the processing "
                            "range of the program, please re-enter...")

        coordinate_li.append([temp, pos_ratio])

    data_df = pd.DataFrame(coordinate_li, columns=["bin_value", "pos_ratio"])

    # It is necessary to add (0, 0) and (1, 1) to the data set,
    # which is equivalent to adding two boxes, so that the
    # measurable range of the isotonic regression is [0,1]
    data_df = data_df.append({"bin_value": 0,
                              "pos_ratio": 0}, ignore_index=True)
    data_df = data_df.append({"bin_value": 1,
                              "pos_ratio": 1}, ignore_index=True)

    # In order to train the isotonic regression model, we need
    # to change bin value sort from small to large
    data_df.sort_values(by="bin_value", inplace=True)

    # Training isotonic regression model,
    # increasing=True means incremental fitting
    iso_reg = IsotonicRegression(increasing=True)
    iso_reg.fit(X=data_df["bin_value"], y=data_df["pos_ratio"])

    # The prediction probability of isotonic regression is calculated
    input_df["iso_pred"] = iso_reg.predict(input_df[proba_name])

    # Determine whether there are polynomials to be fitted
    if is_poly:

        # The additional polynomials are fitted and the predicted
        # results of polynomials are calculated on the total data
        if fit_num > 9 or fit_num <= 0:
            raise Exception("Sorry, the number of fitting "
                            "times you entered exceeds the "
                            "threshold value of 9 or is non "
                            "negative. Please re-enter...")
        elif fit_num > 0:
            for i in sm.range(fit_num):
                z = np.polyfit(data_df["bin_value"],
                               data_df["pos_ratio"], i + 1)
                p = np.poly1d(z)
                colname = "{}_polynomial".format(i + 1)
                input_df[colname] = p(input_df["iso_pred"])

    return input_df


def gaussian_calibrate(input_df, proba_name):
    """Gaussian calibration.

    Parameters
    ----------
    input_df : DataFrame
        There are two columns of DataFrame,
        one is prediction probability, the other is label value.

    proba_name : str
        Prediction probability name.

    Returns
    --------
    res : DataFrame
        DataFrame composed of prediction
        probability and calibration probability.

    Examples
    ---------
    >>> test_df
            label     proba
    1           1  0.241217
    2           0  0.096250
    3           0  0.140861
    4           0  0.119471
    5           0  0.005491
    ...       ...       ...
    149996      0  0.019120
    149997      0  0.097704
    149998      0  0.020250
    149999      0  0.111851
    150000      0  0.024276
    >>> res = mag_calibrate.gaussian_calibrate(test_df, proba_name="proba")
    >>> res
            label     proba  gauss_pred
    1           1  0.241217    0.737106
    2           0  0.096250    0.569554
    3           0  0.140861    0.631961
    4           0  0.119471    0.603528
    5           0  0.005491    0.195389
    ...       ...       ...         ...
    149996      0  0.019120    0.378461
    149997      0  0.097704    0.571756
    149998      0  0.020250    0.386301
    149999      0  0.111851    0.592524
    150000      0  0.024276    0.409872
    """

    # Generate standard normal distribution
    # random numbers and sort them from small to large
    norm_rand = sorted(np.random.normal(0, 1, input_df.shape[0]))

    # Get the maximum and minimum values
    max_val, min_val = max(norm_rand), min(norm_rand)

    # Normalize the normal data to the range of [0,1]
    norm_stand = (norm_rand - min_val) / (max_val - min_val)

    # Ranking according to prediction probability
    input_df.sort_values(by=proba_name, inplace=True)

    # Increase the prediction probability of a gaussian calibaration
    input_df["gauss_pred"] = norm_stand

    # Sort according to the row index to ensure
    # that the input and output tags are in the same order
    input_df.sort_index(inplace=True)

    return input_df


def score_calibrate(input_df, proba_name, min_score=300, max_score=850):
    """Score calibration.

    Parameters
    ----------
    input_df : DataFrame
        There are two columns of DataFrame, one is prediction
        probability, the other is label value.

    proba_name : str
        Prediction probability name.

    min_score : float
        Minimum value of calibration score range.

    max_score : float
        Maximum value of calibration score range.

    Returns
    --------
    input_df : DataFrame
        The calculation results of score calibration are added.

    Example
    --------
    >>> test_df
            label     proba
    1           1  0.241217
    2           0  0.096250
    3           0  0.140861
    4           0  0.119471
    5           0  0.005491
    ...       ...       ...
    149996      0  0.019120
    149997      0  0.097704
    149998      0  0.020250
    149999      0  0.111851
    150000      0  0.024276
    >>> res = mag_calibrate.score_calibrate(test_df, proba_name="proba")
    >>> res
            label     proba  score_pred
    1           1  0.241217  436.452929
    2           0  0.096250  354.447409
    3           0  0.140861  379.682812
    4           0  0.119471  367.583115
    5           0  0.005491  303.106043
    ...       ...       ...         ...
    149996      0  0.019120  310.815820
    149997      0  0.097704  355.269631
    149998      0  0.020250  311.455028
    149999      0  0.111851  363.272707
    150000      0  0.024276  313.732501
    """

    # 获取预测概率中的最大最小值
    max_value, min_value = max(input_df[proba_name]), min(input_df[proba_name])

    # 对归一化后的数据进行评分转换
    input_df["score_pred"] = input_df[proba_name].apply(
        lambda x: min_score + (max_score - min_score
                               ) * (x - min_value) / (max_value - min_value))

    return input_df
