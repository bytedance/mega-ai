# coding=utf-8
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from magellan_ai.ml.mag_util import mag_metrics
import math

EPS = 1e-7


def show_func():
    # 可视化函数
    print("-----------------------------------------")
    print("|analyse methods                        |")
    print("-----------------------------------------")
    print("|feature_coverage_in_diff_people        |")
    print("|single_enum_feat_eval_diff_people      |")
    print("|single_continuity_feat_eval_diff_people|")
    print("-----------------------------------------")


def feature_coverage_in_diff_people(
        df_,
        group_col,
        group_dict={},
        col_no_cover_dict={},
        col_handler_dict={},
        cols_skip=[],
        is_sorted=True):
    """
    对于两人群根据样本特征列计算特征覆盖率
    Parameters:
        df_: DataFrame
            输入文件
        group_col:
            表示人群的标签所在的列
        group_dict : dict
            group 标签的别名，不出现在此 dict 的标签，别名等于自身
        col_no_cover_dict: dict
            自定义特征指定数据类型的非覆盖值. 默认值:
                * int64: [0, -1]
                * float64: [0.0, -1.0]
                * object: []
                * bool: []
        col_handler_dict: dict
            指定特征数据类型的覆盖率计算方法.
        cols_skip:
            忽略计算特征覆盖率的特征名称 .
        is_sorted: bool
            是否对特征覆盖率进行排序
    """
    # 默认非覆盖标签
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}
    # 取出两人群个特征信息
    groups = df.groupby(group_col)
    group_dfs, indexs = [], []
    for index, group_df in groups:
        if index in col_no_cover_dict[str(df_[group_col].dtype)]:
            continue
        indexs.append(index)
        group_dfs.append(group_df)
    # 人群种类数不为 2，抛出异常
    try:
        if len(group_dfs) != 2:
            raise Exception("人群种类数不为 2")
    except Exception as err:
        print(err)
    # 调用 metrics 库，生成两人群标签覆盖率 DataFrame
    df1 = mag_metrics.cal_feature_coverage(
        group_dfs[0],
        col_no_cover_dict,
        col_handler_dict,
        cols_skip,
        is_sorted)
    df2 = mag_metrics.cal_feature_coverage(
        group_dfs[1],
        col_no_cover_dict,
        col_handler_dict,
        cols_skip,
        is_sorted)

    # 修改两个 coverage 列信息
    df1.columns = ['feature', 'coverage_%s' % indexs[0], 'feat_type']
    df2.columns = ['feature', 'coverage_%s' % indexs[1], 'feat_type']
    del df1['feat_type']
    # 合并 DataFrame
    res_df = pd.merge(df1, df2, how="inner", on="feature")

    return res_df


"""
# 1. 分析两人群在单个枚举类特征上差异。特征的例子：性别，职业，最喜欢的 app
# 2. 特征取值为 2 种时，进行卡方检验，特征大于 2 种时，计算 psi 值。其它统计方法后续可以添加。
"""


def single_enum_feat_eval_diff_people(
        group_df_col,
        feature_df_col,
        group_dict={},
        feature_dict={},
        col_no_cover_dict={},
        draw_pics=False):
    """
    Parameters:
        group_df_col : Series
            人群包所在的列
        feature_df_col : Series
            需要分析的特征所在的列
        group_dict : dict
            group 标签的别名，不出现在此 dict 的标签，别名等于自身
        feature_dict : dict
            feature 标签别名，不出现在此 dict 的标签，别名等于自身
        col_no_cover_dict : dict
            非覆盖特征字典
        draw_pics : bool
            是否需要绘制图片，默认为 False
    Returns:
        report : dict
            "DataFrame": DataFrame
                两人群均值方差
            "p-value": float
                t-test / Welch 检验结果
            "result": str
                差异性的结论

    Notes:
        卡方检验中 H0：π1=π2，H1：π1≠π2，
        卡方检验自由度为 1，卡方值 >= 3.841，p-value <= 0.05 可以认为有显著差异。
    """
    # 默认非覆盖标签
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}

    # 合并 group_df_col, featutre_df_col, 删掉非覆盖标签所在的行，结果保存在 df 中
    group_list, feature_list = [], []
    for group, feature in zip(group_df_col, feature_df_col):
        # 保留 group 和 feature 同时有效的信息
        if group not in col_no_cover_dict[str(
                group_df_col.dtype)] and feature not in col_no_cover_dict[
                    str(feature_df_col.dtype)]:
            group_list.append(group)
            feature_list.append(feature)
            # 把不在的 dict 的标签别名设为自身
            if group not in group_dict:
                group_dict[group] = group
            if feature not in feature_dict:
                feature_dict[feature] = feature

    df = pd.DataFrame({'group': group_list, 'feature': feature_list})

    # 人群种类，特征种类
    group_counter = len(df.groupby("group"))
    feature_counter = len(df.groupby('feature'))

    # 异常处理，特征种类数少于 2，人群种类数不等于 2 时，抛出异常
    try:
        if feature_counter <= 1:
            raise Exception("此标签特征种类小于 2，无法差异分析")
    except Exception as err:
        print(err)
    try:
        if group_counter != 2:
            raise Exception("人群种类数不为 2")
    except Exception as err:
        print(err)

    # 计算各人群中各 feature 个数与比例
    counter, ratio = {}, {}  # 人群标签向其特征信息的映射
    for group_tag in group_dict:
        tmp_counter, tmp_ratio = {}, {}  # group_tag 人群中，特征标签向其出现次数比例的映射
        # 计算 group_tag 人群中 feature_tag 出现次数
        total = df.groupby("group").get_group(
            group_tag).shape[0]  # group_tag 人数
        for feature_tag in feature_dict:
            cnt = df.groupby("group").get_group(group_tag).groupby(
                "feature").get_group(feature_tag).shape[0]
            tmp_counter[feature_tag] = cnt
            tmp_ratio[feature_tag] = 1.0 * cnt / total
        counter[group_tag] = tmp_counter
        ratio[group_tag] = tmp_ratio

    if feature_counter == 2:
        # 卡方检验
        # 生成 dataFrame，两人群中两种标签出现次数与占比
        group_tag0, group_tag1 = group_dict.keys()  # 各 group 标签
        group0, group1 = group_dict.values()  # 各 group 标签别名
        feature0, feature1 = feature_dict.values()  # 每种 feature 标签的别名
        cnt0, cnt1 = counter[group_tag0], counter[group_tag1]  # 两个组中，每个特征:出现几次
        ratio0, ratio1 = ratio[group_tag0], ratio[group_tag1]  # 两个组中，每个特征:组内占比
        res_df = pd.DataFrame({'features': [feature0, feature1],
                               "%s 人数" % group0: list(cnt0.values()),
                               "%s 比例" % group0: list(ratio0.values()),
                               "%s 人数" % group1: list(cnt1.values()),
                               "%s 比例" % group1: list(ratio1.values())})
        # 卡方计算
        a, b, c, d = list(cnt0.values()) + list(cnt1.values())
        n = a + b + c + d
        chi2 = 1.0 * n * (a * d - b * c) * (a * d - b * c) / \
            (a + b) / (a + c) / (b + d) / (c + d)
        # 绘制 barh
        if draw_pics:
            index = [group0, group1]
            df_barh1 = pd.DataFrame({feature0: [cnt0[0], cnt1[0]], feature1: [
                                    cnt0[1], cnt1[1]]}, index=index)
            df_barh1.plot.barh(stacked=True)
            df_barh2 = pd.DataFrame(
                {feature0: [ratio0[0], ratio1[0]], feature1: [
                                    ratio0[1], ratio1[1]]}, index=index)
            df_barh2.plot.barh(stacked=True)
        # 生成报告
        report = {"DataFrame": res_df, "chi2": chi2}
        report["result"] = "根据卡方检验，两人群分布无明显差异" \
            if chi2 <= 3.841 else "根据卡方检验，两人群分布有明显差异"
        return report
    else:
        # psi 计算
        def calpsi(a, e):
            e = max(e, 0.0005)
            a = max(a, 0.0005)
            return (a - e) * math.log(a / e)

        portion0, portion1 = ratio.values()  # 两个人群各种特征出现频率
        group0, group1 = group_dict.values()  # 各 group 标签别名

        # 如果portion0的分组中key没有落入portion1，那么对应portion1的取值为0
        for x in portion0:
            if x not in portion1:
                portion1[x] = 0

        # 如果portion0的分组中key没有落入portion1，那么对应portion1的取值为0
        for x in portion1:
            if x not in portion0:
                portion0[x] = 0

        # 分别将两组人群变成数据框并根据特征值进行拼接
        group0_df = pd.DataFrame.from_dict(
            portion0,
            orient="index",
            columns=[group0]).reset_index().rename(
            columns={
                "index": "feat_value"})
        group1_df = pd.DataFrame.from_dict(
            portion1,
            orient="index",
            columns=[group1]).reset_index().rename(
            columns={
                "index": "feat_value"})
        res_df = pd.merge(group0_df, group1_df, how="inner", on="feat_value")

        # psi 计算
        psi_li = []  # psi List
        psi_sum = 0
        for i in range(len(res_df)):
            psi_value = calpsi(res_df[group0][i], res_df[group1][i])
            psi_sum += psi_value
            psi_li.append(psi_value)

        # 生成显示标签在人群中占比的 DataFrame，并按差异大小，对特征标签排序
        res_df["psi"] = psi_li
        res_df = res_df.sort_values(by=['psi'], ascending=False)
        res_df.index = [i for i in range(res_df.shape[0])]

        # 绘制两人群饼图
        if draw_pics:
            limit = 25
            if len(portion0) <= limit:
                df0_draw = pd.DataFrame(
                    {group0: list(portion0.values())}, index=list(
                        portion0.keys()))
                _ = df0_draw.plot.pie(y=group0, figsize=(5, 5))

                df1_draw = pd.DataFrame(
                    {group1: list(portion1.values())}, index=list(
                        portion1.keys()))
                _ = df1_draw.plot.pie(y=group1, figsize=(5, 5))
            else:
                print("此特征种类数太多，不适合绘制饼图")

        # 生成报告
        report = {"DataFrame": res_df, "psi": psi_sum}
        if psi_sum < 0.1:
            report["result"] = "差异很小"
        elif psi_sum < 0.25:
            report["result"] = "有一定差异"
        else:
            report["result"] = "有较大差异"

        return report


"""
1. 从均值分析两人群中，较为连续的特征差异（例如：概率类标签）
2. 绘制两人群 hist
3. 目前支持 t-test / Welch ，效果不佳，后续检验方法需要更新。
"""


def single_continuity_feat_eval_diff_people(
        group_df_col,
        feature_df_col,
        group_dict={},
        col_no_cover_dict={},
        draw_pics=False):
    """
    Parameters:
        group_df_col : Series
            人群包所在的列
        feature_df_col : Series
            需要分析的特征所在的列
        group_dict : dict
            group 标签的别名，不出现在此 dict 的标签，别名等于自身
        col_no_cover_dict : dict
            非覆盖特征字典
        draw_pics : bool
            是否需要绘制图片，默认为 False
    Returns:
        report : dict
            "DataFrame": DataFrame
                两人群均值方差
            "p-value": float
                t-test / Welch 检验结果
            "result": str
                差异性的结论

    Notes:
        H0 两分布均值相同，H1 两分布均值不同，此函数根据两人群方差、样本数来选择 t-test / Welch 检验
    """
    # 默认非覆盖
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}
    # 合并 group_df_col, featutre_df_col, 删掉非覆盖标签所在的行，结果保存在 df 中
    group_list, feature_list = [], []
    for group, feature in zip(group_df_col, feature_df_col):
        # group 和 feature 同时被覆盖才保留
        if group not in col_no_cover_dict[str(
                group_df_col.dtype)] and feature > -EPS:
            group_list.append(group)
            feature_list.append(feature)
            # 将不在 group_dict 中的别名设为自身
            if group not in group_dict:
                group_dict[group] = group

    group0, group1 = list(group_dict.values())
    df = pd.DataFrame({'group': group_list, 'feature': feature_list})

    # 计算人群种类数
    group_counter = len(df.groupby("group"))
    # 当人群种类数不等于 2 时，抛出异常
    try:
        if group_counter != 2:
            raise Exception("人群种类数不为 2")
    except Exception as err:
        print(err)

    # 当 feature 为 obj 时，抛出异常
    try:
        if str(df['feature'].dtype) == "object":
            raise Exception("fearute 为 object")
    except Exception as err:
        print(err)

    # 计算两个人群包均值与方差，并生成 DataFrame
    groups = df.groupby('group')
    vec0 = groups.get_group(0)['feature']
    vec1 = groups.get_group(1)['feature']
    features = ['mean', 'std']
    info0 = [vec0.mean(), vec0.std()]  # 计算第一组均值、方差
    info1 = [vec1.mean(), vec1.std()]  # 计算第二组均值、方差
    res_df = pd.DataFrame(
        data={'features': features, group0: info0, group1: info1})

    # 绘制分布柱状图
    if draw_pics:
        plt.hist(vec0, bins=100, color="#FF0000", alpha=.9), plt.title(
            "%s" % group0), plt.show()
        plt.hist(vec1, bins=100, color="#FF0000", alpha=.9), plt.title(
            "%s" % group1), plt.show()

    # t 检验 / Welch 检验
    L = stats.ttest_ind(vec0, vec1, equal_var=True if stats.levene(
        vec0, vec1).pvalue <= 0.05 else False)

    # 生成结论报告
    report = {"DataFrame": res_df, "p-value": L.pvalue}
    report["result"] = "有较大差异" if L.pvalue < 0.05 else "无明显差异"
    return report


if __name__ == '__main__':
    # TEST CODE
    """
    df = pd.read_csv("./car.csv", header=0)
    print(feature_coverage_in_diff_people(
        df,"car4",group_dict={0:"ins", 1:"water"}))
    """

    """
    df = pd.read_csv("./car.csv", header=0)
    dic = single_enum_feat_eval_diff_people(
        df['car4'], df['own_car'], group_dict={
            0: "ins", 1: "water"}, feature_dict={
            0: "not own car", 1: "own car"}, draw_pics=True)
    print(dic['DataFrame'])
    print("chi2=%s, result=%s" % (dic["chi2"], dic["result"]))
    df = pd.read_csv("./t3.csv", header=0)
    dic = single_enum_feat_eval_diff_people(
        df['t3_4'], df['career'], group_dict={
            0: "ins", 1: "water"}, draw_pics=True)
    print(dic['DataFrame'])
    print("psi=%s, result=%s" % (dic["psi"], dic["result"]))
    """

    df = pd.read_csv('./prob.csv', header=0)
    dic = single_continuity_feat_eval_diff_people(
        df["prob4"], df["proba_parenting"], group_dict={
            0: "ins", 1: "water"}, draw_pics=True)
    print(dic['DataFrame'])
    print("p-value=%s, result=%s" % (dic["p-value"], dic["result"]))
