import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def show_func():
    print("feature evaluation methods")
    print("1.cal_iv")
    print("2.cal_feature_coverage")
    print("<--------------------------->")
    print("model evaluation methods")
    print("1.cal_auc")
    print("2.cal_ks")
    print("3.cal_psi")
    print("4.cal_lift")
    print("<--------------------------->")
    print("supervised binning methods")
    print("1.chiSquare_binning_boundary")
    print("2.decisionTree_binning_boundary")


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

    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # rank = [label_value for prob_value, label_value in
    #         sorted(zip(y_pred, y_true), key=lambda x: x[0])]
    # rank_list = [index + 1 for index, label_value in
    #              enumerate(rank) if label_value == 1]
    # pos_num, neg_num = \
    #     len(y_true[y_true == 1]), len(y_true[y_true == 0])
    # auc = (sum(rank_list) -
    #        (pos_num * (pos_num + 1)) / 2) / (pos_num * neg_num)

    return roc_auc_score(y_true, y_pred)


def cal_lift(y_true, y_pred, k_part=10):
    """Calculate lift
    :param y_true: 标签值
    :param y_pred: 预测概率
    :param k_part: 最大分箱个数
    :return: lift, depth, thresholds
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 根据最大分箱个数确定阈值列表
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
    for i in range(len(thres_list)):
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
    """Calculate PSI
    :param base_score: 训练集上的预测概率
    :param cur_score: 测试集上的预测概率
    :param k_part: 最大分箱个数
    :return: PSI
    """

    # 根据k_part获取分箱间隔值列表
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

    # 计算每组的psi值
    for i in range(len(thres_list[:-1])):
        base_rate = sum((np.array(base_score) > thres_list[i])
                        & (np.array(base_score) <=
                           thres_list[i + 1])) / base_score_len
        cur_rate = sum((np.array(cur_score) > thres_list[i])
                       & (np.array(cur_score) <=
                          thres_list[i + 1])) / cur_score_len
        psi += (base_rate - cur_rate) * np.log(base_rate / (cur_rate + 0.0001))

    return psi


def cal_iv(df_, label_name, is_sorted=True, k_part=10,
           bin_method="same_frequency"):
    """ 计算IV值
    :param df_: 样本集
    :param label_name: 标签名
    :param is_sorted: 针对iv值是否排序
    :param k_part: 最大的分箱个数
    :param bin_method: 分箱类型，目前提供了等频分箱，卡方分箱以及决策树分箱
    :return: iv_df
    """

    def get_ivi(input_df, label_name, pos_num, neg_num):

        posi_num, negi_num = sum(input_df[label_name] == 1), \
                             sum(input_df[label_name] == 0)
        posri, negri = (posi_num + 0.0001) * 1.0 / pos_num, \
                       (negi_num + 0.0001) * 1.0 / neg_num
        woei = np.log(posri / negri)
        ivi = (posri - negri) * np.log(posri / negri)
        return ivi, woei

    # 提取特征列表
    df_.fillna(0, inplace=True)
    feat_list = list(df_.columns)
    feat_list.remove(label_name)

    # 计算每个特征的iv值
    iv_dict = {}
    pos_num, neg_num = sum(df_[label_name] == 1), sum(df_[label_name] == 0)
    for col_name in feat_list:

        iv_total = 0
        cur_feat_woe = {}

        # 将类别特征变成数值特征
        if df_[col_name].dtypes in (np.dtype('bool'), np.dtype('object')):
            label_encoder = {label: idx for idx, label
                             in enumerate(np.unique(df_[col_name]))}
            df_[col_name] = df_[col_name].map(label_encoder)

        # 等频分箱
        if bin_method == "same_frequency":

            # 如果特征取值个数小于默认的分组个数，那么直接按照枚举值进行分组
            if len(df_[col_name].unique()) < k_part:
                boundary_list = [df_[col_name].min() -
                                 0.001] + [df_[col_name].unique().sort()]
            else:
                cur_feat_interval = sorted(
                    pd.qcut(df_[col_name], k_part, duplicates="drop").unique())
                boundary_list = [cur_feat_interval[0].left] + \
                                [value.right for value in cur_feat_interval]
        # 决策树分箱
        elif bin_method == "decision_tree":
            boundary_list = decisionTree_binning_boundary(
                df_, col_name, label_name, k_part)

        # 卡方分箱
        elif bin_method == "chi_square":

            # 如果特征的取值个数大于100，那么需要先将其等频离散化成100个值
            if len(df_[col_name].unique()) >= 100:
                cur_feat_interval = \
                    pd.qcut(df_[col_name], 100, duplicates="drop")
                df_[col_name] = cur_feat_interval

                # 根据划分区间左右端点的平均数作为离散的枚举值，将连续特征转成离散特征
                df_[col_name] = df_[col_name].apply(
                    lambda x: float((x.left + x.right) / 2))

            boundary_list = chiSquare_binning_boundary(
                df_, col_name, label_name, k_part)
            df_[col_name] = df_[col_name].astype("float64")

        else:
            raise Exception("The current {} method is not "
                            "implemented".format(bin_method))

        for i in range(len(boundary_list) - 1):
            cur_group_df = df_[(df_[col_name] > boundary_list[i])
                               & (df_[col_name] <= boundary_list[i + 1])]
            interval = "(" + str(boundary_list[i]) + ", " \
                       + str(boundary_list[i + 1]) + "]"
            ivi, woei = get_ivi(cur_group_df, label_name, pos_num, neg_num)
            cur_feat_woe[interval] = woei
            iv_total += ivi

        iv_dict[col_name] = [cur_feat_woe, iv_total]

    iv_df = pd.DataFrame.from_dict(
        iv_dict, orient="index", columns=["woe_value", "iv_value"])
    iv_df = iv_df.reset_index().rename(columns={"index": "feature"})
    if is_sorted:
        iv_df.sort_values(by="iv_value", inplace=True,
                          ascending=False, ignore_index=True)
    return iv_df


def cal_feature_coverage(df_, col_no_cover_dict={},
                         col_handler_dict={}, cols_skip=[], is_sorted=True):
    """analyze feature coverage for pandas dataframe

    :param df_: 输入数据
    :param col_no_cover_dict: 自定义特征指定数据类型的非覆盖值. 默认值:
            * int64: [0, -1]
            * float64: [0.0, -1.0]
            * object: []
            * bool: []
    :param col_handler_dict: 指定特征数据类型的覆盖率计算方法.
    :param cols_skip: 忽略计算特征覆盖率的特征名称 .
    :param is_sorted: 是否对特征覆盖率进行排序
    :return:
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

    row_num = df_.shape[0]
    feat_coverage_dict = {}
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

    feat_coverage_df = pd.DataFrame.from_dict(
        feat_coverage_dict, orient="index",
        columns=["coverage", "feat_type"], )
    feat_coverage_df = feat_coverage_df\
        .reset_index().rename(columns={"index": "feature"})

    if is_sorted:
        feat_coverage_df.sort_values(
            by="coverage", inplace=True, ascending=False, ignore_index=True)

    return feat_coverage_df


# 卡方分箱
def chiSquare_binning_boundary(df_, feat_name, label_name, k_part):
    """
    :param df_: 输入数据框
    :param feat_name: 特征名称
    :param label_name: 标签名称
    :param k_part: 最大分箱个数
    :return: boundary: 分箱的边界列表
    """

    all_num = df_.shape[0]
    pos_num = df_[label_name].sum()
    expected_ratio = pos_num / all_num
    feat_value_list = sorted(df_[feat_name].unique())

    # 计算每个区间的chi2统计量
    chi2_list = []
    pos_list = []
    expected_pos_list = []

    for feat_value in feat_value_list:
        temp_pos_num = df_.loc[df_[feat_name] ==
                               feat_value, label_name].sum()
        temp_all_num = df_.loc[df_[feat_name] ==
                               feat_value, label_name].count()

        expected_pos_num = temp_all_num * expected_ratio
        chi2_value = (temp_pos_num -
                      expected_pos_num) ** 2 / expected_pos_num
        chi2_list.append(chi2_value)
        pos_list.append(temp_all_num)
        expected_pos_list.append(expected_pos_num)

    # 将结果导出到数据框
    chi2_df = pd.DataFrame({feat_name: feat_value_list,
                            "chi2_value": chi2_list,
                            "pos_num": pos_list,
                            "expected_pos_cnt": expected_pos_list})

    # 根据index合并chi2_df中相邻位置的数值
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

    # 计算当前特征的卡方分箱的个数，即当前特征的所有枚举值的个数
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


# 决策树分箱
def decisionTree_binning_boundary(df_, feat_name, label_name, k_part):
    """
    :param df_: 输入数据框
    :param feat_name: 特征名称
    :param label_name: 标签名称
    :param k_part: 最大分箱个数
    :return: boundary: 分箱的边界列表
    """

    # 存储分箱的边界值
    boundary = []
    label_value = df_[label_name].values
    feat_value = df_[feat_name].values.reshape(-1, 1)

    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_leaf_nodes=k_part,
                                 min_samples_leaf=0.05)
    clf.fit(feat_value, label_value)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for index in range(n_nodes):
        if children_left[index] != children_right[index]:
            boundary.append(threshold[index])
    boundary.sort()
    min_x = feat_value.min() - 0.0001
    max_x = feat_value.max()
    boundary = [min_x] + boundary + [max_x]

    return boundary


if __name__ == "__main__":
    # 测试代码
    show_func()
    # data_df = pd.read_csv("/Users/bytedance/ByteCode"
    #                       "/magellan_ai/data/cs-training.csv", index_col=0)
    # data_df.fillna(0, inplace=True)
    #
    # res = chiSquare_binning_boundary(
    #     data_df, "NumberOfTime30-59DaysPastDueNotWorse",
    #     "SeriousDlqin2yrs", 10)
    # res2 = decisionTree_binning_boundary(
    #     data_df, "NumberOfTime30-59DaysPastDueNotWorse",
    #     "SeriousDlqin2yrs", 10)
    # print(res)

    # df_t1 = pd.read_csv('/Users/bytedance/ByteCode/magellan_ai/data/t1.csv')
    # df_t2 = pd.read_csv('/Users/bytedance/ByteCode/magellan_ai/data/t2.csv')
    # res = cal_psi(df_t1.loc[:, 'multiloan_online_model_v8_score'],
    #               df_t2.loc[:, 'multiloan_online_model_v8_score'], k_part=10)
    # print(res)
