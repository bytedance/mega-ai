import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from mag_metrics import chiSquare_binning_boundary, \
    decisionTree_binning_boundary

"""
    Author: huangning.honey@bytedance.com
    Date: 2020/08/04
    Function: 实现不同的模型分数校准方法
"""


def show_calibrate_func():
    print("1.isotonic_calibrate")
    print("2.gaussian_calibrate")
    print("3.score_calibrate")


def isotonic_calibrate(input_df, proba_name, label_name,
                       is_poly=False, fit_num=3, bin_num=1000,
                       bin_method="same_frenquency", bin_value_method="mean"):
    """保序回归校准
    :param input_df: 输入数据框，一列是预测概率，一列是标签值
    :param proba_name: 数据框中的预测概率名称
    :param label_name: 数据框中的标签名称
    :param is_poly: 是否针对分箱结果进行多项式拟合，用于和保序回归结果进行比较
    :param fit_num: 多项式拟合次数
    :param bin_num: 分箱个数
    :param bin_method: 分箱方法
    {'same_frequency','decision_tree','chi_square'}，默认是等频分箱
    :param bin_value_method: 箱值计算方法："mean"
    :return: input_df 新增了保序回归的计算结果(多项式拟合结果)
    """

    if bin_method == "same_frequency":

        # 如果特征取值个数小于默认的分组个数，那么直接按照枚举值进行分组
        if len(input_df[proba_name].unique()) < bin_num:
            boundary_list = [input_df[proba_name].min() -
                             0.001] + [input_df[proba_name].unique().sort()]
        else:
            cur_feat_interval = sorted(
                pd.qcut(input_df[proba_name], bin_num,
                        precision=3, duplicates="drop").unique())

            boundary_list = [cur_feat_interval[0].left] + \
                            [value.right for value in cur_feat_interval]

    # 决策树分箱
    elif bin_method == "decision_tree":
        boundary_list = decisionTree_binning_boundary(
            input_df[proba_name], input_df[label_name], bin_num)

    # 卡方分箱
    elif bin_method == "chi_square":

        # 如果特征枚举值个数大于100，需要先将其等频离散成100个值, 否则计算速度会很慢
        if len(input_df[proba_name].unique()) >= 100:
            cur_feat_interval = \
                pd.qcut(input_df[proba_name], 100, duplicates="drop")
            input_df[proba_name] = cur_feat_interval

            # 根据划分区间左右端点的平均数作为离散的枚举值，实现将连续特征转成离散特征
            input_df[proba_name] = input_df[proba_name].apply(
                lambda x: float((x.left + x.right) / 2))

        boundary_list = chiSquare_binning_boundary(
            input_df, proba_name, label_name, bin_num)
        input_df[proba_name] = input_df[proba_name].astype("float64")

    else:
        raise Exception("The current {} method is not "
                        "implemented".format(bin_method))

    # 二维坐标点，第1维是桶值，第2维是真实的通过率
    coordinate_li = []
    for i in range(len(boundary_list) - 1):

        group_df = input_df[(input_df[proba_name] > boundary_list[i])
                            & (input_df[proba_name] <= boundary_list[i + 1])]

        # 计算当前组的正例占比
        pos_ratio = sum(group_df[label_name] == 1) / group_df.shape[0] \
            if group_df.shape[0] != 0 else 1

        if group_df.shape[0] == 0:

            # 如果当前组没有元素，默认取右侧端点
            temp = boundary_list[i + 1]
        elif bin_value_method == "mean":
            temp = group_df[proba_name].mean()
        elif bin_value_method == "medium":
            temp = group_df[proba_name].median()
        elif bin_value_method == "mode":

            # 众数可能存在多个，默认取其中第一个
            temp = group_df[proba_name].mode()[0]
        else:
            raise Exception("bin_value_method entered "
                            "is not within the processing "
                            "range of the program, please re-enter...")

        coordinate_li.append([temp, pos_ratio])

    data_df = pd.DataFrame(coordinate_li, columns=["bin_value", "pos_ratio"])

    # 必须向数据集中添加（0，0）和（1，1），相当于增加两个分箱，使得保序模型的横坐标可测范围是从[0,1]
    data_df = data_df.append({"bin_value": 0,
                              "pos_ratio": 0}, ignore_index=True)
    data_df = data_df.append({"bin_value": 1,
                              "pos_ratio": 1}, ignore_index=True)

    # 为了后面的保序回归模型，需要将bin_val从小到大排序
    data_df.sort_values(by="bin_value", inplace=True)

    # 训练保序回归模型：increasing=True表示进行增拟合
    iso_reg = IsotonicRegression(increasing=True)
    iso_reg.fit(X=data_df["bin_value"], y=data_df["pos_ratio"])

    # 计算全量样本的保序回归预测概率
    input_df["iso_pred"] = iso_reg.predict(input_df[proba_name])

    # 判断是否还有多项式需要拟合
    if is_poly:

        # 拟合额外的多项式, 并在全量数据上计算多项式的预测结果
        if fit_num > 9 or fit_num <= 0:
            raise Exception("Sorry, the number of fitting "
                            "times you entered exceeds the "
                            "threshold value of 9 or is non "
                            "negative. Please re-enter...")
        elif fit_num > 0:
            for i in range(fit_num):
                z = np.polyfit(data_df["bin_value"],
                               data_df["pos_ratio"], i + 1)
                p = np.poly1d(z)
                colname = "{}_polynomial".format(i + 1)
                input_df[colname] = p(input_df["iso_pred"])

    return input_df


def gaussian_calibrate(input_df, proba_name):
    """高斯校准
    :param input_df: 输入数据框
    :param proba_name: 预测概率名称
    :return: res: 预测概率和校准概率的数据框
    """
    # 生成标准正态分布随机数,并从小到大排序
    norm_rand = sorted(np.random.normal(0, 1, input_df.shape[0]))

    # 获取最大值和最小值
    max_val, min_val = max(norm_rand), min(norm_rand)

    # 对正态数据进行最大最小归一化,变成【0，1】的rdd
    norm_stand = (norm_rand - min_val) / (max_val - min_val)

    # 按照预测概率进行排序
    input_df.sort_values(by=proba_name, inplace=True)

    # 增加一列高斯分布的预测概率
    input_df["gauss_pred"] = norm_stand

    # 根据行索引进行排序保证输入和输出的标签顺序一致
    input_df.sort_index(inplace=True)

    return input_df


def score_calibrate(input_df, proba_name, min_score=300, max_score=850):
    """得分校准
    :param input_df: 输入数据框，一列是预测概率proba，一列是标签值label
    :param proba_name: 数据框中预测概率的名称
    :param min_score: 校准分数范围的最小值
    :param max_score: 校准分数范围的最大值
    :return: input_df 新增一列分数校准
    """

    # 获取预测概率中的最大最小值
    max_value, min_value = max(input_df[proba_name]), min(input_df[proba_name])

    # 对归一化后的数据进行评分转换
    input_df["score_pred"] = input_df[proba_name].apply(
        lambda x: min_score + (max_score - min_score
                               ) * (x - min_value) / (max_value - min_value))

    return input_df


if __name__ == "__main__":
    data_df = pd.read_csv("/Users/bytedance/ByteCode/magellan_ai"
                          "/data/cs-training.csv", index_col=0)
    data_df.fillna(0, inplace=True)
    X = data_df.iloc[:, 1:]
    y = data_df["SeriousDlqin2yrs"]
    lr = LogisticRegression(penalty="l2", random_state=99)
    lr.fit(X, y)
    y_proba = lr.predict_proba(X)[:, 1]
    test_df = pd.DataFrame({"label": y, "proba": y_proba})

    # # 测试show_calibrate_func
    # show_calibrate_func()

    # 测试isotonic_calibrate
    res1 = isotonic_calibrate(
        test_df, proba_name="proba", label_name="label",
        is_poly=True, bin_method="chi_square", bin_value_method="mean")
    res1.sort_values(by="proba", inplace=True)

    # # 测试score_calibrate
    # res2 = score_calibrate(test_df, proba_name="proba")
    # print(res2)
    #
    # res3 = gaussian_calibrate(test_df, "proba")
    # print(res3)