# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import six.moves as sm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def show_func():
    print("+-------------------------------+")
    print("|feature evaluation methods     |")
    print("|1.cal_iv                       |")
    print("|2.cal_feature_coverage         |")
    print("+-------------------------------+")
    print("|model evaluation methods       |")
    print("|1.cal_auc                      |")
    print("|2.cal_ks                       |")
    print("|3.cal_psi                      |")
    print("|4.cal_lift                     |")
    print("+-------------------------------+")
    print("|supervised binning methods     |")
    print("|1.chiSquare_binning_boundary   |")
    print("|2.decisionTree_binning_boundary|")
    print("+-------------------------------+")


def cal_ks(y_true, y_pred):
    """Calculate KS.

    Parameters
    ----------
    y_true : list, ndarray, Series or DataFrame
        True label, the value range is {0, 1}.

    y_pred : list, ndarray, Series or DataFrame
        Prediction probability, the value range is [0, 1].

    Returns
    --------
    cutoff : float
        The threshold corresponding to KS value.

    ks : float
        KS value of the model.

    Examples
    ----------
    >>> y_true_li = [1, 1, 0, 1, 0, 0]
    >>> y_pred_li = [0.1, 0.6, 0.3, 0.8, 0.6, 0.2]
    >>> cutoff, ks = mag_metrics.cal_ks(y_true_li, y_pred_li)
    >>> cutoff
    0.2
    >>> ks
    0.33333333333333337

    >>> y_true_arr = np.array([1, 1, 0, 1, 0, 0])
    >>> y_pred_arr = np.array([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> cutoff, ks = mag_metrics.cal_ks(y_true_arr, y_pred_arr)
    >>> cutoff
    0.2
    >>> ks
    0.33333333333333337

    >>> y_true_ser = pd.Series([1, 1, 0, 1, 0, 0])
    >>> y_pred_ser = pd.Series([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> cutoff, ks = mag_metrics.cal_ks(y_true_ser, y_pred_ser)
    >>> cutoff
    0.2
    >>> ks
    0.33333333333333337

    >>> y_true_df = pd.DataFrame([1, 1, 0, 1, 0, 0])
    >>> y_pred_df = pd.DataFrame([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> cutoff, ks = mag_metrics.cal_ks(y_true_df, y_pred_df)
    >>> cutoff
    0.2
    >>> ks
    0.33333333333333337

    Notes
    -----
    The length of real label and prediction probability should be consisten.
    """

    if len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        tpr_fpr_gap = abs(tpr - fpr)
        ks = max(tpr_fpr_gap)
        cutoff = thresholds[tpr_fpr_gap == ks][0]
        return cutoff, ks
    return -1, -1


def cal_auc(y_true, y_pred):
    """Calculate AUC.

    Parameters
    ----------
    y_true : list, ndarray, Series or DataFrame
        True label, the value range is {0, 1}.

    y_pred : list, ndarray, Series or DataFrame
        Prediction probability, the value range is [0, 1].

    Returns
    --------
    auc: float
        AUC value of the model.

    Examples
    ----------
    >>> y_true_li = [1, 1, 0, 1, 0, 0]
    >>> y_pred_li = [0.1, 0.6, 0.3, 0.8, 0.6, 0.2]
    >>> auc = mag_metrics.cal_auc(y_true_li, y_pred_li)
    >>> auc
    0.5555555555555556

    >>> y_true_arr = np.array([1, 1, 0, 1, 0, 0])
    >>> y_pred_arr = np.array([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> auc = mag_metrics.cal_auc(y_true_arr, y_pred_arr)
    >>> auc
    0.5555555555555556

    >>> y_true_ser = pd.Series([1, 1, 0, 1, 0, 0])
    >>> y_pred_ser = pd.Series([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> auc = mag_metrics.cal_auc(y_true_ser, y_pred_ser)
    >>> auc
    0.5555555555555556

    >>> y_true_df = pd.DataFrame([1, 1, 0, 1, 0, 0])
    >>> y_pred_df = pd.DataFrame([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> auc = mag_metrics.cal_auc(y_true_df, y_pred_df)
    >>> auc
    0.5555555555555556

    Notes
    -----
    The length of real label and prediction probability should be consisten.
    """

    if len(set(y_true)) > 1:
        return roc_auc_score(y_true, y_pred)
    return -1


def cal_lift(y_true, y_pred, k_part=10):
    """Calculate lift

    Parameters
    ----------
    y_true : list, ndarray, Series or DataFrame
        True label, the value range is {0, 1}.

    y_pred : list, ndarray, Series or DataFrame
        Prediction probability, the value range is [0, 1].

    k_part : int, default=10
        Maximum number of bins.

    Returns
    ----------
    lift: list
        Lift values under different thresholds.

    depth: list
        Depth values under different thresholds.

    thresholds: list
        List of thresholds.

    Examples
    ----------
    >>> y_true_li = [1, 1, 0, 1, 0, 0]
    >>> y_pred_li = [0.1, 0.6, 0.3, 0.8, 0.6, 0.2]
    >>> lift, depth, thresholds = mag_metrics.cal_lift(y_true_li, y_pred_li)
    >>> lift
    [1, 2.0, 1.3333333333333333, 1.0, 0.8]
    >>> depth
    [0, 0.16666666666666666, 0.5, 0.6666666666666666, 0.8333333333333334]
    >>> thresholds
    [0.8, 0.6, 0.3, 0.2, 0.1]

    >>> y_true_arr = np.array([1, 1, 0, 1, 0, 0])
    >>> y_pred_arr = np.array([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> lift, depth, thresholds = mag_metrics.cal_lift(y_true_arr, y_pred_arr)
    >>> lift
    [1, 2.0, 1.3333333333333333, 1.0, 0.8]
    >>> depth
    [0, 0.16666666666666666, 0.5, 0.6666666666666666, 0.8333333333333334]
    >>> thresholds
    [0.8, 0.6, 0.3, 0.2, 0.1]

    >>> y_true_ser = pd.Series([1, 1, 0, 1, 0, 0])
    >>> y_pred_ser = pd.Series([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> lift, depth, thresholds = mag_metrics.cal_lift(y_true_ser, y_pred_ser)
    >>> lift
    [1, 2.0, 1.3333333333333333, 1.0, 0.8]
    >>> depth
    [0, 0.16666666666666666, 0.5, 0.6666666666666666, 0.8333333333333334]
    >>> thresholds
    [0.8, 0.6, 0.3, 0.2, 0.1]

    >>> y_true_df = pd.DataFrame([1, 1, 0, 1, 0, 0])
    >>> y_pred_df = pd.DataFrame([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> lift, depth, thresholds = mag_metrics.cal_lift(y_true_df, y_pred_df)
    >>> lift
    [1, 2.0, 1.3333333333333333, 1.0, 0.8]
    >>> depth
    [0, 0.16666666666666666, 0.5, 0.6666666666666666, 0.8333333333333334]
    >>> thresholds
    [0.8, 0.6, 0.3, 0.2, 0.1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine the threshold list according to the maximum number of bins
    if len(np.unique(y_pred)) <= k_part:
        thres_list = sorted(np.unique(y_pred), reverse=True)
    else:
        intervals = sorted(np.unique(pd.qcut(y_pred, k_part,
                                             duplicates="drop")), reverse=True)
        thres_list = [interval.right for interval in intervals]

    lift, depth = [], []
    y_pos_pred = y_pred[y_true == 1]
    y_neg_pred = y_pred[y_true == 0]
    y_num = y_true.shape[0]
    pos_rate = y_pos_pred.shape[0] / y_num
    for i in sm.range(len(thres_list)):
        TP, FP = len(y_pos_pred[y_pos_pred > thres_list[i]]), \
                 len(y_neg_pred[y_neg_pred > thres_list[i]])
        if TP + FP == 0:
            lift.append(1)
            depth.append(0)
        else:
            precision = TP / (TP + FP)
            lift.append(precision / pos_rate)
            depth.append((TP + FP) / y_num)

    return lift, depth, thres_list


def cal_psi(base_score, cur_score, k_part=10):
    """Calculate PSI.

    Parameters
    ----------
    base_score : list, ndarray, Series or DataFrame
        The prediction probability on the training set.

    cur_score: list, ndarray, Series or DataFrame
        The Prediction probability on the test set.

    k_part : int, default=10
        Maximum number of bins.

    Returns
    ----------
    psi : float
        PSI value of the model.

    Examples
    ----------
    >>> y_train_pred_li = [0.1, 0.6, 0.3, 0.8, 0.6, 0.2]
    >>> y_test_pred_li = [0.2, 0.3, 0.4, 0.9, 0.2, 0.1]
    >>> psi = mag_metrics.cal_psi(y_train_pred_li, y_test_pred_li)
    >>> psi
    1.4674292331341745

    >>> y_train_pred_arr = np.array([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> y_test_pred_arr = np.array([0.2, 0.3, 0.4, 0.9, 0.2, 0.1])
    >>> psi = mag_metrics.cal_psi(y_train_pred_arr, y_test_pred_arr)
    >>> psi
    1.4674292331341745

    >>> y_test_pred_ser = pd.Series([0.2, 0.3, 0.4, 0.9, 0.2, 0.1])
    >>> psi = mag_metrics.cal_psi(y_train_pred_ser, y_test_pred_ser)
    >>> psi
    1.4674292331341745

    >>> y_train_pred_df = pd.DataFrame([0.1, 0.6, 0.3, 0.8, 0.6, 0.2])
    >>> y_test_pred_df = pd.DataFrame([0.2, 0.3, 0.4, 0.9, 0.2, 0.1])
    >>> psi = mag_metrics.cal_psi(y_train_pred_df, y_test_pred_df)
    >>> psi
    array([1.46742923])
    """

    # Determine the threshold list according to the maximum number of bins
    if len(np.unique(base_score)) <= k_part:
        thres_list = sorted(np.unique(base_score))
        thres_list = [min(thres_list) - 0.001] + thres_list
    else:
        intervals = sorted(np.unique(
            pd.qcut(base_score, k_part, duplicates="drop")), reverse=False)
        thres_list = [intervals[0].left] + \
                     [interval.right for interval in intervals]

    base_score_len = len(base_score)
    cur_score_len = len(cur_score)
    psi = 0

    # Calculate the PSI value of each bin
    for i in sm.range(len(thres_list[:-1])):
        base_rate = sum((np.array(base_score) > thres_list[i])
                        & (np.array(base_score) <=
                           thres_list[i + 1])) / base_score_len
        cur_rate = sum((np.array(cur_score) > thres_list[i])
                       & (np.array(cur_score) <=
                          thres_list[i + 1])) / cur_score_len
        psi += (base_rate - cur_rate) * np.log(base_rate / (cur_rate + 0.0001))

    return psi


def cal_iv(input_df, label_name, is_sorted=True, k_part=10,
           bin_method="same_frequency"):
    """Calculate IV.

    Parameters
    ----------
    input_df : DataFrame
        Sample set.

    label_name : str
        Label name.

    k_part : int, default=10
        Maximum number of bins.

    is_sorted : bool, default=True
        Whether to sort in descending order for IV value.

    k_part : int, default=10
        Maximum number of bins.

    bin_method : {'same_frequency', 'decision_tree', 'chi_square'}, \
                 default='same_frequency'
        binning methods.

    Returns
    ----------
    iv_df : DataFrame
        The IV of the features and the woe value of each group.

    Examples
    ----------
    >>> df
    SeriousDlqin2yrs  ...  NumberOfDependents
    1                      1  ...                 2.0
    2                      0  ...                 1.0
    3                      0  ...                 0.0
    4                      0  ...                 0.0
    5                      0  ...                 0.0
    ...                  ...  ...                 ...
    149996                 0  ...                 0.0
    149997                 0  ...                 2.0
    149998                 0  ...                 0.0
    149999                 0  ...                 0.0
    150000                 0  ...                 0.0
    >>> res = mag_metrics.cal_iv(df, "SeriousDlqin2yrs", is_sorted=True,
    ... k_part=10, bin_type="same_frequency")
    >>> print(res)
                  feature                         woe_value     iv_value
    0  RevolvingUtiliz...  {'(-0.001, 0.00297]': -1.019..., 1.113033e+00
    1  NumberOfTime30-...  {'(-0.001, 1.0]': -0.2578258..., 4.718310e-01
    2                 age  {'(-0.001, 33.0]': 0.5812925..., 2.591578e-01
    3           DebtRatio  {'(-0.001, 0.0309]': -0.2313..., 7.369553e-02
    4  NumberOfOpenCre...  {'(-0.001, 3.0]': 0.51482573..., 6.689181e-02
    5       MonthlyIncome  {'(-0.001, 2358.0]': -0.0049..., 5.821497e-02
    6  NumberOfDependents  {'(-0.001, 1.0]': -0.0882658..., 2.496452e-02
    7  NumberRealEstat...  {'(-0.001, 1.0]': 0.02428457..., 1.209142e-02
    8  NumberOfTimes90...  {'(-0.001, 98.0]': 9.2596487..., 8.574110e-17
    9  NumberOfTime60-...  {'(-0.001, 98.0]': 9.2596487..., 8.574110e-17
    """

    def get_ivi(input_df, label_name, pos_num, neg_num):

        posi_num, negi_num = sum(input_df[label_name] == 1), \
                             sum(input_df[label_name] == 0)
        posri, negri = (posi_num + 0.0001) * 1.0 / pos_num, \
                       (negi_num + 0.0001) * 1.0 / neg_num
        woei = np.log(posri / negri)
        ivi = (posri - negri) * np.log(posri / negri)
        return ivi, woei

    # Extract feature list
    feat_list = list(input_df.columns)
    feat_list.remove(label_name)

    # The woe and IV values of each feature are calculated
    iv_dict = {}
    pos_num, neg_num = sum(input_df[label_name] == 1), sum(
        input_df[label_name] == 0)
    feat_len = len(feat_list)
    for index, col_name in enumerate(feat_list):

        iv_total = 0
        cur_feat_woe = {}

        # Split the non null value and null value part of the current feature
        input_na_df = input_df[input_df[col_name].isna()]
        input_df = input_df[~input_df[col_name].isna()]

        # Turn object features into numerical features
        if input_df[col_name].dtypes in (np.dtype('bool'), np.dtype('object')):
            label_encoder = {label: idx for idx, label
                             in enumerate(np.unique(input_df[col_name]))}
            input_df[col_name] = input_df[col_name].map(label_encoder)

        if bin_method == "same_frequency":

            # If the number of feature enumerations is
            # less than k_part, they are grouped
            # directly according to the
            # enumeration value
            if len(input_df[col_name].unique()) < k_part:
                boundary_list = [input_df[col_name].min() -
                                 0.001] + sorted(input_df[col_name].unique())
            else:
                cur_feat_interval = sorted(
                    pd.qcut(input_df[col_name], k_part,
                            duplicates="drop").unique())
                boundary_list = [cur_feat_interval[0].left] + \
                                [value.right for value in cur_feat_interval]

        elif bin_method == "decision_tree":

            boundary_list = decisiontree_binning_boundary(
                input_df, col_name, label_name, k_part)

        elif bin_method == "chi_square":

            # If the number of values of the feature is greater than 100,
            # it is necessary to discretize the feature into 100 values
            # at the same frequency to speed up the operation
            if len(input_df[col_name].unique()) >= 100:
                cur_feat_interval = \
                    pd.qcut(input_df[col_name], 100, duplicates="drop")
                input_df[col_name] = cur_feat_interval

                # According to the average of the left and right endpoints
                # of the partition interval as the discrete enumeration
                # value, the continuous feature is transformed into
                # the discrete feature
                input_df[col_name] = input_df[col_name].apply(
                    lambda x: float((x.left + x.right) / 2))

            boundary_list = chisquare_binning_boundary(
                input_df, col_name, label_name, k_part)
            input_df[col_name] = input_df[col_name].astype("float64")

        else:
            raise Exception("The current {} method is not "
                            "implemented".format(bin_method))

        for i in sm.range(len(boundary_list) - 1):
            cur_group_df = input_df[(input_df[col_name] > boundary_list[i])
                                    & (input_df[col_name]
                                       <= boundary_list[i + 1])]
            interval = "(" + str(boundary_list[i]) + ", " \
                       + str(boundary_list[i + 1]) + "]"

            ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
            cur_feat_woe[interval] = woei
            iv_total += ivi

        # The IVI and woei of the missing values were calculated separately
        if input_na_df.shape[0] != 0:
            cur_group_df = input_na_df
            ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
            cur_feat_woe["NaN"] = woei
            iv_total += ivi

        iv_dict[col_name] = [cur_feat_woe, iv_total]
        print("\rFeature IV calculation compl"
              "eted {:.2%}".format((index+1)/feat_len), end="")

    print()
    iv_df = pd.DataFrame.from_dict(
        iv_dict, orient="index", columns=["woe_value", "iv_value"])
    iv_df = iv_df.reset_index().rename(columns={"index": "feature"})
    if is_sorted:
        iv_df.sort_values(by="iv_value", inplace=True,
                          ascending=False, ignore_index=True)
    return iv_df


def cal_feature_coverage(input_df, col_no_cover_dict={},
                         col_handler_dict={}, cols_skip=[], is_sorted=True):
    """Analyze feature coverage for pandas dataframe.

    Parameters
    ----------
    input_df : DataFrame
        Sample set.

    col_no_cover_dict : dict
        A custom feature specifies a non overriding value for the data type.

    col_handler_dict : dict
        A custom feature specifies a Coverage calculation
        method for the data type.

    cols_skip : list
        Ignore feature names for which feature coverage is calculated.


    is_sorted : bool
        Whether to arrange feature coverage in reverse order.

    Returns
    ----------
    feat_coverage_df : DataFrame
        Feature coverage.

    Examples
    ----------
    >>> df
            SeriousDlqin2yrs  ...  NumberOfDependents
    1                      1  ...                 2.0
    2                      0  ...                 1.0
    3                      0  ...                 0.0
    4                      0  ...                 0.0
    5                      0  ...                 0.0
    ...                  ...  ...                 ...
    149996                 0  ...                 0.0
    149997                 0  ...                 2.0
    149998                 0  ...                 0.0
    149999                 0  ...                 0.0
    150000                 0  ...                 0.0
    >>> ans = mag_metrics.cal_feature_coverage(df,
    ... cols_skip=["SeriousDlqin2yrs"])
    >>> print(ans)
                                    feature  coverage feat_type
    0                                   age  0.999993     int64
    1       NumberOfOpenCreditLinesAndLoans  0.987413     int64
    2                             DebtRatio  0.972580   float64
    3  RevolvingUtilizationOfUnsecuredLines  0.927480   float64
    4                         MonthlyIncome  0.790900   float64
    5          NumberRealEstateLoansOrLines  0.625413     int64
    6                    NumberOfDependents  0.394493   float64
    7  NumberOfTime30-59DaysPastDueNotWorse  0.159880     int64
    8               NumberOfTimes90DaysLate  0.055587     int64
    9  NumberOfTime60-89DaysPastDueNotWorse  0.050693     int64
    """

    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [0, -1],
                             'float64': [0.0, -1.0], 'object': [], 'bool': []}

    def col_handler_bool(df_col):
        return df_col.isna().sum()

    def col_handler_object(df_col):
        return df_col.isna().sum()

    def col_handler_int64(df_col):

        row_cnt = 0
        for col_aim_value in col_no_cover_dict['int64']:
            row_cnt += df_col[df_col == col_aim_value].shape[0]

        return row_cnt + df_col.isna().sum()

    def col_handler_float64(df_col):
        row_cnt = 0
        for col_aim_value in col_no_cover_dict['float64']:
            row_cnt = row_cnt + df_col[
                abs(df_col - col_aim_value) <= 1e-6].shape[0]

        return row_cnt + df_col.isna().sum()

    row_num = input_df.shape[0]
    feat_coverage_dict = {}
    cols_len = len(input_df.columns)

    for index, col_name in enumerate(input_df.columns):
        if col_name in cols_skip:
            continue

        col_handler = col_handler_int64
        if input_df[col_name].dtype == np.dtype('bool'):
            if 'bool' in col_handler_dict:
                col_handler = col_handler_dict['bool']
            else:
                col_handler = col_handler_bool

        if input_df[col_name].dtype == np.dtype('object'):
            if 'object' in col_handler_dict:
                col_handler = col_handler_dict['object']
            else:
                col_handler = col_handler_object

        if input_df[col_name].dtype == np.dtype('int64'):
            if 'int64' in col_handler_dict:
                col_handler = col_handler_dict['int64']
            else:
                col_handler = col_handler_int64

        if input_df[col_name].dtype == np.dtype('float64'):
            if 'float64' in col_handler_dict:
                col_handler = col_handler_dict['float64']
            else:
                col_handler = col_handler_float64

        no_cover_count = col_handler(input_df.loc[:, col_name])
        coverage = (row_num - no_cover_count) * 1.0 / (row_num + 1e-6)

        feat_coverage_dict[col_name] = [coverage, input_df[col_name].dtype]
        print("\rFeature coverage calculation "
              "completed {:.2%}".format((index+1)/cols_len), end="")

    print()
    feat_coverage_df = pd.DataFrame.from_dict(
        feat_coverage_dict, orient="index",
        columns=["coverage", "feat_type"], )
    feat_coverage_df = feat_coverage_df \
        .reset_index().rename(columns={"index": "feature"})

    if is_sorted:
        feat_coverage_df.sort_values(
            by="coverage", inplace=True, ascending=False, ignore_index=True)

    return feat_coverage_df


# 卡方分箱
def chisquare_binning_boundary(input_df, feat_name, label_name, k_part):
    """Calculate binning threshold list by chisqure binning method.

    Parameters
    ----------
    input_df : DataFrame
        Sample set.

    feat_name : str
        Feature name.

    label_name : str
        Label name.

    k_part : int
        Maximum number of bins.

    Returns
    ---------
    boundary : list
        Boundary list of bins.

    Examples
    ----------
    >>> df
            SeriousDlqin2yrs  ...  NumberOfDependents
    1                      1  ...                 2.0
    2                      0  ...                 1.0
    3                      0  ...                 0.0
    4                      0  ...                 0.0
    5                      0  ...                 0.0
    ...                  ...  ...                 ...
    149996                 0  ...                 0.0
    149997                 0  ...                 2.0
    149998                 0  ...                 0.0
    149999                 0  ...                 0.0
    150000                 0  ...                 0.0
    >>> res = mag_metrics.chisquare_binning_boundary(df,
    ... "NumberOfTime30-59DaysPastDueNotWorse", "SeriousDlqin2yrs", 10)
    >>> res
    [-0.0001, 0, 1, 2, 3, 4, 5, 6, 7, 96, 98]
    """

    all_num = input_df.shape[0]
    pos_num = input_df[label_name].sum()
    expected_ratio = pos_num / all_num
    feat_value_list = sorted(input_df[feat_name].unique())

    # Calculate the Chi2 statistic for each interval
    chi2_list = []
    pos_list = []
    expected_pos_list = []

    for feat_value in feat_value_list:
        temp_pos_num = input_df.loc[input_df[feat_name] ==
                                    feat_value, label_name].sum()
        temp_all_num = input_df.loc[input_df[feat_name] ==
                                    feat_value, label_name].count()

        expected_pos_num = temp_all_num * expected_ratio
        chi2_value = (temp_pos_num -
                      expected_pos_num) ** 2 / expected_pos_num
        chi2_list.append(chi2_value)
        pos_list.append(temp_all_num)
        expected_pos_list.append(expected_pos_num)

    # Export results to a DataFrame
    chi2_df = pd.DataFrame({feat_name: feat_value_list,
                            "chi2_value": chi2_list,
                            "pos_num": pos_list,
                            "expected_pos_cnt": expected_pos_list})

    # Merge chi2_df according to index values of adjacent positions
    def merge(input_df, merge_index, origin_index):

        input_df.loc[merge_index, "pos_num"] = \
            input_df.loc[merge_index, "pos_num"] \
            + input_df.loc[origin_index, "pos_num"]
        input_df.loc[merge_index, "expected_pos_cnt"] = \
            input_df.loc[merge_index, "expected_pos_cnt"] \
            + input_df.loc[origin_index, "expected_pos_cnt"]
        input_df.loc[merge_index, "input_value"] = \
            (input_df.loc[merge_index, "pos_num"] -
             input_df.loc[merge_index, "expected_pos_cnt"]) ** 2 \
            / input_df.loc[merge_index, "expected_pos_cnt"]
        input_df.drop(origin_index, axis=0, inplace=True)
        input_df.reset_index(drop=True, inplace=True)

        return input_df

    # Calculate the number of chi square bins of the current feature,
    # that is, the number of all enumeration values of the current feature
    group_num = len(chi2_df)
    while group_num > k_part:
        min_index = chi2_df[chi2_df["chi2_value"]
                            == chi2_df["chi2_value"].min()].index[0]
        if min_index == 0:
            chi2_df = merge(chi2_df, min_index + 1, min_index)
        elif min_index == group_num - 1:
            chi2_df = merge(chi2_df, min_index, min_index - 1)
        else:
            if chi2_df.loc[min_index - 1, "chi2_value"] \
                    > chi2_df.loc[min_index + 1, "chi2_value"]:
                chi2_df = merge(chi2_df, min_index + 1, min_index)
            else:
                chi2_df = merge(chi2_df, min_index, min_index - 1)

        group_num = len(chi2_df)

    min_x = chi2_df[feat_name].min() - 0.0001
    boundary = [min_x] + list(chi2_df[feat_name])
    return boundary


def decisiontree_binning_boundary(input_df, feat_name, label_name, k_part):
    """Calculate binning threshold list by decisionTree binning method.

    Parameters
    ----------
    input_df : DataFrame
        Sample set.

    feat_name : str
        Feature name.

    label_name : str
        Label name.

    k_part : int
        Maximum number of bins.

    Returns
    ---------
    boundary : list
        Boundary list of bins.

    Examples
    ---------
    >>> df
            SeriousDlqin2yrs  ...  NumberOfDependents
    1                      1  ...                 2.0
    2                      0  ...                 1.0
    3                      0  ...                 0.0
    4                      0  ...                 0.0
    5                      0  ...                 0.0
    ...                  ...  ...                 ...
    149996                 0  ...                 0.0
    149997                 0  ...                 2.0
    149998                 0  ...                 0.0
    149999                 0  ...                 0.0
    150000                 0  ...                 0.0
    >>> res = mag_metrics.decisiontree_binning_boundary(df,
    ... "NumberOfTime30-59DaysPastDueNotWorse", "SeriousDlqin2yrs", 10)
    >>> res
    [-0.0001, 0.5, 1.5, 98]
    """

    # Store the boundary value of the bin
    boundary = []
    label_value = input_df[label_name].values
    feat_value = input_df[feat_name].values.reshape(-1, 1)

    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_leaf_nodes=k_part,
                                 min_samples_leaf=0.05)
    clf.fit(feat_value, label_value)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for index in sm.range(n_nodes):
        if children_left[index] != children_right[index]:
            boundary.append(threshold[index])
    boundary.sort()
    min_x = feat_value.min() - 0.0001
    max_x = feat_value.max()
    boundary = [min_x] + boundary + [max_x]

    return boundary
