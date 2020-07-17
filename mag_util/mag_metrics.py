import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


"""
    Author: huangning.honey
    Date: 2020/07/13
    Function: Integrating common model evaluation methods and feature evaluation indexes
"""


def show_func():

    print("Feature evaluation methods:")
    print("cal_iv, cal_feature_coverage")
    print("<==========================================>")
    print("Model evaluation methods:")
    print("cal_auc, cal_ks, cal_psi, cal_lift")


def cal_ks(y_true, y_pred):
    """
    Calculate KS
    :param y_true: True label value
    :param y_pred: Prediction probability
    :return: cutoff, ks
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    tpr_fpr_gap = abs(fpr - tpr)
    ks = max(tpr_fpr_gap)
    cutoff = thresholds[tpr_fpr_gap == ks][0]
    return cutoff, ks


def cal_auc(y_true, y_pred):

    """Calculate AUC
    params:
    * y_true: 真实标签值 {0,1}
    * y_pred: 模型预测概率 [0,1]
    """

    rank = [label_value for prob_value, label_value in sorted(zip(y_pred, y_true), key=lambda x: x[0])]
    rank_list = [index + 1 for index, label_value in enumerate(rank) if label_value == 1]
    pos_num, neg_num = len(y_true[y_true == 1]), len(y_true[y_true == 0])
    auc = (sum(rank_list) - (pos_num * (pos_num + 1)) / 2) / (pos_num * neg_num)

    return auc


def cal_lift(y_true, y_pred, k_part=10):
    """Calculate lift
    :param y_true: True label value
    :param y_pred: Prediction probability
    :param k_part:
    :return: lift, depth, thresholds
    """

    y_df = pd.DataFrame({
        "y_pred": y_pred,
        "y_true": y_true
    })

    # Determine threshold list
    if len(y_df.y_pred.unique()) <= k_part:
        thres_list = sorted(y_df.y_pred.unique(), reverse=True)
    else:
        interval_df = pd.qcut(y_df.y_pred, k_part, duplicates="drop")
        y_df["interval_right"] = pd.Series(interval_df).apply(lambda x: x.right)
        thres_list = sorted(y_df["interval_right"].unique(), reverse=True)

    lift, depth, thresholds = [], [], []
    P = y_df[y_df.y_true == 1].y_pred
    N = y_df[y_df.y_true == 0].y_pred
    y_num = len(y_df)
    p_rate = len(P)/y_num
    for threshold in thres_list[1:]:
        TP, FN, FP, TN = len(P[P >= threshold]), len(P[P < threshold]),  len(N[N >= threshold]), len(N[N <= threshold])
        precision = TP/(TP+FP)
        lift.append(precision/p_rate)
        depth.append((TP + FP)/y_num)

    # Add the lift and depth values of the left boundary
    lift.insert(0, 1)
    depth.insert(0, 0)

    return lift, depth, thresholds


def cal_psi(base_score, cur_score, k_part=10):
    """Calculate PSI
    :param base_score: Prediction probability of model on training set
    :param cur_score: Prediction probability of model on testing set
    :param k_part: Number of groups
    :return: PSI
    """

    # According to the prediction probability, the box is divided
    interval_series = pd.qcut(base_score, k_part, duplicates="drop")
    interval_sorted = sorted(pd.Series(interval_series).unique())
    interval_list = [(inter.left, inter.right) for inter in interval_sorted]

    # The PSI value of each group was accumulated
    base_score_len = len(base_score)
    cur_score_len = len(cur_score)
    psi = 0
    for left, right in interval_list:
        base_rate = sum((np.array(base_score) > left) & (np.array(base_score) <= right))/base_score_len
        cur_rate = sum((np.array(cur_score) > left) & (np.array(cur_score) <= right))/cur_score_len
        psi += (base_rate - cur_rate) * np.log(base_rate/(cur_rate+0.0001))

    return psi


def cal_iv(df_, label_name, is_sorted=True, k_part=10, bin_type="same_frequency"):

    """ Calculate iv
    :param df_: input data
    :param label_name: the name of label
    :param is_sorted: Sort for IV value
    :param k_part: number of buckets
    :param bin_type: frequency or chiSquare
    :return: iv_df
    """

    def get_ivi(input_df, label_name, pos_num, neg_num):

        posi_num, negi_num = sum(input_df[label_name] == 1), sum(input_df[label_name] == 0)
        posri, negri = (posi_num + 0.0001) * 1.0 / pos_num, (negi_num + 0.0001) * 1.0 / neg_num
        woei = np.log(posri / negri)
        ivi = (posri - negri) * np.log(posri / negri)
        return ivi, woei

    # Extract feature list
    df_.fillna(0, inplace=True)
    feat_list = list(df_.columns)
    feat_list.remove(label_name)

    # Calculate the IV value of the every feature
    iv_dict = {}
    pos_num, neg_num = sum(df_[label_name] == 1), sum(df_[label_name] == 0)
    for col_name in feat_list:

        iv_total = 0
        cur_feat_woe = {}

        # Determine the number of groups
        if len(df_[col_name].unique()) < k_part:
            for feat_value in df_[col_name].unique():
                cur_group_df = df_[df_[col_name] == feat_value]
                ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
                cur_feat_woe[[feat_value]] = woei
                iv_total += ivi
        else:

            # Change category variables into numerical variables
            if df_[col_name].dtypes in (np.dtype('bool'), np.dtype('object')):
                label_encoder = {label: idx for idx, label in enumerate(np.unique(df_[col_name]))}
                df_[col_name] = df_[col_name].map(label_encoder)

            if bin_type == "chiSquare":
                if len(df_[col_name].unique()) >= 100:
                    print("current feature '{}' is not a category feature, so bin_type of current feature is "
                          "automatically converted to 'same_frequency'".format(col_name))

                    cur_feat_interval = pd.qcut(df_[col_name], k_part, duplicates="drop").unique()
                    for interval in cur_feat_interval:
                        cur_group_df = df_[(df_[col_name] > interval.left) & (df_[col_name] <= interval.right)]
                        ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
                        cur_feat_woe[interval] = woei
                        iv_total += ivi
                else:
                    chi2_obj = Chi2Binning()
                    chi2_df = chi2_obj.cal_chi2_value(df_, col_name, label_name)
                    cur_feat_interval = chi2_obj.chi2_merge_interval(chi2_df, k_part)[col_name]
                    for i in range(len(cur_feat_interval)):
                        if i == 0:
                            cur_group_df = df_[(df_[col_name] <= cur_feat_interval[i])]
                            interval = "(-inf, " + str(i) + "]"
                        else:
                            cur_group_df = df_[(df_[col_name] > cur_feat_interval[i-1]) & (df_[col_name] <= cur_feat_interval[i])]
                            interval = "(" + str(i-1) + ", " + str(i) + "]"
                        ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
                        cur_feat_woe[interval] = woei
                        iv_total += ivi

            # select a binning method
            elif bin_type == "same_frequency":
                # Calculate the grouping of each value for the current feature
                cur_feat_interval = pd.qcut(df_[col_name], k_part, duplicates="drop").unique()
                for interval in cur_feat_interval:
                    cur_group_df = df_[(df_[col_name] > interval.left) & (df_[col_name] <= interval.right)]
                    ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
                    cur_feat_woe[interval] = woei
                    iv_total += ivi
            else:
                print("The current method {} is not implemented".format(bin_type))
                return
        iv_dict[col_name] = [cur_feat_woe, iv_total]

    iv_df = pd.DataFrame.from_dict(iv_dict, orient="index", columns=["woe_value", "iv_value"])
    iv_df = iv_df.reset_index().rename(columns={"index": "feature"})
    if is_sorted:
        iv_df.sort_values(by="iv_value", inplace=True, ascending=False, ignore_index=True)
    return iv_df


def cal_feature_coverage(df_, col_no_cover_dict={}, col_handler_dict={}, cols_skip=[], is_sorted=True):
    """analyze feature coverage for pandas dataframe

    :param df_: input data
    :param col_no_cover_dict: a dict for custom no-cover value. by default:
            * int64: [0, -1]
            * float64: [0.0, -1.0]
            * object: []
            * bool: []
    :param col_handler_dict: a dict for custom non-cover value statistics.
    :param cols_skip: a list for skip cols which will not analyze the coverage.
    :param is_sorted: Sort by coverage
    :return:
    """

    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [0, -1], 'float64': [0.0, -1.0], 'object': [], 'bool': []}

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
            row_cnt = row_cnt + df_col[abs(df_col - col_aim_value) <= 1e-6].shape[0]

        return row_cnt + df_col.isna().sum()

    row_num, col_num = df_.shape[0], df_.shape[1]
    feat_coverage_dict = {}
    feat_dtype_dict = {}
    for col_name in df_.columns:
        if col_name in cols_skip:
            continue

        col_handler = col_handler_int64
        if df_[col_name].dtype == np.dtype('bool'):
            if 'bool' in col_handler_dict:
                col_handler = col_handler_dict['bool']
            else:
                col_handler = col_handler_bool

        if df_[col_name].dtype == np.dtype('object'):
            if 'object' in col_handler_dict:
                col_handler = col_handler_dict['object']
            else:
                col_handler = col_handler_object

        if df_[col_name].dtype == np.dtype('int64'):
            if 'int64' in col_handler_dict:
                col_handler = col_handler_dict['int64']
            else:
                col_handler = col_handler_int64

        if df_[col_name].dtype == np.dtype('float64'):
            if 'float64' in col_handler_dict:
                col_handler = col_handler_dict['float64']
            else:
                col_handler = col_handler_float64

        no_cover_count = col_handler(df_.loc[:, col_name])
        coverage = (row_num - no_cover_count) * 1.0 / (row_num + 1e-6)

        feat_coverage_dict[col_name] = [coverage, df_[col_name].dtype]

    feat_coverage_df = pd.DataFrame.from_dict(feat_coverage_dict, orient="index", columns=["coverage", "feat_type"], )
    feat_coverage_df = feat_coverage_df.reset_index().rename(columns={"index": "feature"})

    if is_sorted:
        feat_coverage_df.sort_values(by="coverage", inplace=True, ascending=False, ignore_index=True)

    return feat_coverage_df


class Chi2Binning:
    """
        Chi square box
    """

    # calculate chi2 value of every binning of the select feature
    def cal_chi2_value(self, df_, feat_name, label_name):

        pos_num = df_[label_name].sum()
        all_num = df_.shape[0]
        expected_ratio = pos_num/all_num

        # Arrange a feature value from small to large
        df_.dropna(inplace=True)
        feat_value_list = sorted(df_[feat_name].unique())

        # cal chi2 statistic in every interval
        chi2_list = []
        pos_list = []
        expected_pos_list = []

        for feat_value in feat_value_list:

            temp_pos_num = df_.loc[df_[feat_name] == feat_value, label_name].sum()
            temp_all_num = df_.loc[df_[feat_name] == feat_value, label_name].count()

            expected_pos_num = temp_all_num * expected_ratio
            chi2_value = (temp_pos_num - expected_pos_num) ** 2 / expected_pos_num
            chi2_list.append(chi2_value)
            pos_list.append(temp_all_num)
            expected_pos_list.append(expected_pos_num)

        # Export results to dataframe
        chi2_df = pd.DataFrame({feat_name: feat_value_list, "chi2_value": chi2_list, "pos_num": pos_list, "expected_pos_cnt": expected_pos_list})
        return chi2_df


    # control the max group num less than interval
    def chi2_merge_interval(self, chi2_df, dfree=4, significance_level=0.1, max_interval=5):

        group_num = len(chi2_df)
        while group_num > max_interval:
            min_index = chi2_df[chi2_df["chi2_value"] == chi2_df["chi2_value"].min()].index[0]
            if min_index == 0:
                chi2_df = self.merge(chi2_df, min_index+1, min_index)
            elif min_index == group_num-1:
                chi2_df = self.merge(chi2_df, min_index, min_index-1)
            else:
                if chi2_df.loc[min_index-1, "chi2_value"] > chi2_df.loc[min_index + 1, "chi2_value"]:
                    chi2_df = self.merge(chi2_df, min_index+1, min_index)
                else:
                    chi2_df = self.merge(chi2_df, min_index, min_index-1)
            group_num = len(chi2_df)

        return chi2_df

    # The minimum chi square value should be greater than or equal to the quantile corresponding to the significance
    # level (so as to ensure that the probability of the first type of error < = alpha), or the number of intervals
    # of merge is less than the specified number
    def chi2_merge(self, chi2_df, dfree=4, significance_level=0.1, max_interval=5):

        quantile = chi2.isf(q=significance_level, df=dfree)
        min_chi2_value = chi2_df["chi2_value"].min()
        group_num = len(chi2_df)

        while min_chi2_value < quantile and group_num > max_interval:

            # find the index of min chi2 value
            min_index = chi2_df[chi2_df["chi2_value"] == chi2_df["chi2_value"].min()].index[0]

            if min_index == 0:
                chi2_df = self.merge(chi2_df, min_index+1, min_index)
            elif min_index == group_num-1:
                chi2_df = self.merge(chi2_df, min_index, min_index-1)
            else:
                if chi2_df.loc[min_index-1, "chi2_value"] > chi2_df.loc[min_index + 1, "chi2_value"]:
                    chi2_df = self.merge(chi2_df, min_index+1, min_index)
                else:
                    chi2_df = self.merge(chi2_df, min_index, min_index-1)
            group_num = len(chi2_df)
        return chi2_df

# merge two chi2_value through index
    def merge(self, chi2_df, merge_index, origin_index):

        chi2_df.loc[merge_index, "pos_num"] = chi2_df.loc[merge_index, "pos_num"] + chi2_df.loc[origin_index, "pos_num"]
        chi2_df.loc[merge_index, "expected_pos_cnt"] = chi2_df.loc[merge_index, "expected_pos_cnt"] + chi2_df.loc[origin_index, "expected_pos_cnt"]
        chi2_df.loc[merge_index, "chi2_value"] = (chi2_df.loc[merge_index, "pos_num"] - chi2_df.loc[merge_index, "expected_pos_cnt"])**2 / chi2_df.loc[merge_index, "expected_pos_cnt"]
        chi2_df.drop(origin_index, axis=0, inplace=True)
        chi2_df.reset_index(drop=True, inplace=True)

        return chi2_df


if __name__ == "__main__":

    # test code
    res = pd.read_csv("/Users/bytedance/Coding/Test/data/cs-training.csv", index_col=0)
    res.fillna(0, inplace=True)

    # y = res.iloc[:, 0]
    # X = res.iloc[:, 1:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # y_train_pred = lr.predict_proba(X_train)[:, 1]
    # y_test_pred = lr.predict_proba(X_test)[:, 1]
    # # show_func()
    print()
    print("cal_iv: ")
    print(cal_iv(res, "SeriousDlqin2yrs", is_sorted=True, k_part=10, bin_type="chiSquare"))

    # print("cal_coverage: ")
    # print(cal_feature_coverage(res))
    # print()
    # print("cal_auc: ", cal_auc(y_train, y_train_pred))
    # print("cal_ks: cutoff={}, ks={}".format(cal_ks(y_train, y_train_pred)[0], cal_ks(y_train, y_train_pred)[1]))
    # print("cal_psi: ", cal_psi(y_train_pred, y_test_pred))
    # print("cal_lift: ", cal_lift(y_train, y_train_pred)[0])