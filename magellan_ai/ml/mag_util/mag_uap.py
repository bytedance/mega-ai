# coding=utf-8
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import math
EPS = 1e-7


def analyse_cover_rate(df, group_col, group0, group1,
                       feature_list=[], col_no_cover_dict={}):
    """ 计算两个人群中，各特征覆盖率与覆盖人数
    Parameters:
        df: DataFrame
            样本集
        group_col: Series
            表示人群的标签所在的列
        group0: str
            0 表示的人群
        group1: str
            1 表示的人群
        feature_list: list
            需要求覆盖率的标签，不传则会统计所有标签覆盖率
        col_no_cover_dict: list
            自定义特征指定数据类型的非覆盖值
    Returns:
        res_df : DataFrame
            两人群中特征的覆盖率以及相应的数据类型
    """
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}

    def col_handler_bool(df_col):
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

    def col_handler_str(df_col):
        row_cnt = 0
        for col_aim_value in col_no_cover_dict['str']:
            row_cnt += df_col[df_col == col_aim_value].shape[0]
        return row_cnt + df_col.isna().sum()

    def col_handler_object(df_col):
        row_cnt = 0
        for col_aim_value in col_no_cover_dict['object']:
            row_cnt += df_col[df_col == col_aim_value].shape[0]
        return row_cnt + df_col.isna().sum()
    # 枚举需要统计的标签，分别计算在两个人群中的覆盖率，并生成 DataFrame
    features, ratio0, cnt0, tot0, ratio1, cnt1, tot1, feat_type = [
    ], [], [], [], [], [], [], []
    for col_name in df.columns:
        if feature_list and col_name not in feature_list:
            continue
        col_handler = col_handler_object
        if df[col_name].dtype == np.dtype('bool'):
            col_handler = col_handler_bool

        if df[col_name].dtype == np.dtype('int64'):
            col_handler = col_handler_int64

        if df[col_name].dtype == np.dtype('float64'):
            col_handler = col_handler_float64

        if df[col_name].dtype == np.dtype('str'):
            col_handler = col_handler_str

        df0 = df.groupby(group_col).get_group(0)
        no_cover0 = col_handler(df0.loc[:, col_name])
        rownum0 = df0.shape[0]
        cover_ratio0 = (1.0 * rownum0 - no_cover0) / rownum0

        df1 = df.groupby(group_col).get_group(1)
        no_cover1 = col_handler(df1.loc[:, col_name])
        rownum1 = df1.shape[0]
        cover_ratio1 = (1.0 * rownum1 - no_cover1) / rownum1
        rownum1 = df1.shape[0]

        features.append(col_name)
        tot0.append(rownum0)
        cnt0.append(rownum0-no_cover0)
        ratio0.append(cover_ratio0)
        tot1.append(rownum1)
        cnt1.append(rownum1-no_cover1)
        ratio1.append(cover_ratio1)
        feat_type.append(df[col_name].dtype)

    res_df = pd.DataFrame(data={'features': features,
                                '%s 总人数' % group0: tot0,
                                '%s 覆盖人数' % group0: cnt0,
                                "%s 中覆盖人数占比" % group0: ratio0,
                                '%s 总人数' % group1: tot1,
                                '%s 覆盖人数' % group1: cnt1,
                                "%s 中覆盖人数占比" % group1: ratio1,
                                'type': feat_type})
    return res_df


def analyse_01(group_df_col, group0, group1,
               feature_df_col, tag0, tag1, draw_pics=False, statistics="chi2"):
    """
    分析两个人群包在二元分类的标签上差异，目前支持卡方检验。
    Parameters:
        group_df_col : Series
            人群包所在的列
        group0 : str
            0 表示的人群
        group1 : str
            1 表示的人群
        feature_df_col : Series
            需要分析的特征所在的列
        tag0 : str
            feature_df_col 中 0 表示的含义
        tag1 : str
            feature_df_col 中 0 表示的含义
        draw_pics : bool
            是否需要绘制图片，默认为 False
        statistics : str
            统计量，目前支持卡方值 "chi2"
    Returns:
        res_df : DataFrame
            两人群中 tag0，tag1 出现次数与占比
        chi2 : float64
            当 statistics 等于 "chi2" 时返回卡方值

    Notes:
        卡方检验中 H0：π1=π2，H1：π1≠π2，
        卡方检验自由度为 1，卡方值 >= 10.828，p-value <= 0.001 可以认为有显著差异。
    """
    df = pd.DataFrame(list(zip(group_df_col, feature_df_col)))
    df.columns = ['group', 'feature']

    # 分别计算两人群中 tag0，tag1 出现次数与占比，并生成一个 DataFrame
    groups = df.groupby('group')
    features, col0, ratio0, col1, ratio1 = [], [], [], [], []
    features = [tag0, tag1]
    g0 = groups.get_group(0)
    col0 = [g0[g0.feature == 0].shape[0], g0[g0.feature == 1].shape[0]]
    g1 = groups.get_group(1)
    col1 = [g1[g1.feature == 0].shape[0], g1[g1.feature == 1].shape[0]]
    ratio0 = [1.0 * col0[0] / (col0[0] + col0[1]), 1.0 * col0[1] /
              (col0[0] + col0[1])] if col0[0] + col0[1] > EPS else [None, None]
    ratio1 = [1.0 * col1[0] / (col1[0] + col1[1]), 1.0 * col1[1] /
              (col1[0] + col1[1])] if col1[0] + col1[1] > EPS else [None, None]
    res_df = pd.DataFrame(data={'features': features, '%s' % group0: col0,
                                '%s比率' % group0: ratio0,
                                '%s' % group1: col1, '%s比率' % group1: ratio1})

    # 绘制图像
    if draw_pics is True:
        index = [group0, group1]
        df_barh1 = pd.DataFrame({tag0: [col0[0], col1[0]], tag1: [
                                col0[1], col1[1]]}, index=index)
        df_barh1.plot.barh(stacked=True)
        df_barh2 = pd.DataFrame({tag0: [ratio0[0], ratio1[0]], tag1: [
                                ratio0[1], ratio1[1]]}, index=index)
        df_barh2.plot.barh(stacked=True)

    # 计算 chi2
    if statistics == "chi2":
        a = col0[0]
        b = col0[1]
        c = col1[0]
        d = col1[1]
        n = a+b+c+d
        chi2 = 1.0 * n * (a*d-b*c) * (a*d-b*c) / (a+b) / (a+c) / (b+d) / (c+d)
        return res_df, chi2

    print("暂未支持此统计量查询")
    return res_df


def analyse_hist(group_df_col, group0, group1,
                 feature_df_col, draw_pics=False, statistics="t-test"):
    """
    分析两个人群包在整数/浮点标签上差异，目前支持 t-test。
    Parameters:
        group_df_col : Series
            人群包所在的列
        group0 : str
            0 表示的人群
        group1 : str
            1 表示的人群
        feature_df_col : Series
            需要分析的特征所在的列
        draw_pics : bool
            是否需要绘制图片，默认为 False
        statistics : str
            统计量，目前支持 "t-test"
    Returns:
        res_df: DataFrame
            两个人群包均值与方差
        pvalue: float64
            t-test 的 pvalue
    Notes:
        pvalue <= 0.001 时，我们可以认为两人群在该特征上有较大差异。
    """
    df = pd.DataFrame(list(zip(group_df_col, feature_df_col)))
    df.columns = ['group', 'feature']
    # 计算两个人群均值与方差，并生成 DataFrame
    groups = df.groupby('group')
    g0 = groups.get_group(0)
    vec0 = g0[g0.feature > - EPS]['feature']
    g1 = groups.get_group(1)
    vec1 = g1[g1.feature > - EPS]['feature']
    features = ['mean', 'std']
    info0 = [vec0.mean(), vec0.std()]
    info1 = [vec1.mean(), vec1.std()]
    res_df = pd.DataFrame(
        data={'features': features, group0: info0, group1: info1})

    # 绘制分布图像
    if draw_pics is True:
        plt.hist(vec0, bins=100, color="#FF0000", alpha=.9), plt.title(
            "%s" % group0), plt.show()
        plt.hist(vec1, bins=100, color="#FF0000", alpha=.9), plt.title(
            "%s" % group1), plt.show()

    # t-test
    if statistics == "t-test":
        L = stats.ttest_ind(vec0, vec1, equal_var=True if stats.levene(
            vec0, vec1).pvalue <= 0.05 else False)
        return res_df, L.pvalue

    print("暂未支持此统计量查询")
    return res_df


def analyse_pie(group_df_col, group0, group1,
                feature_df_col, col_no_cover_dict={},
                draw_pics=False, statistics="psi"):
    """
    分析两个人群包在枚举类标签上差异，目前支持 psi。
    Parameters:
        group_df_col : Series
            人群包所在的列
        group0 : str
            0 表示的人群
        group1 : str
            1 表示的人群
        feature_df_col : Series
            需要分析的特征所在的列
        col_no_cover_dict: list
            自定义特征指定数据类型的非覆盖值
        draw_pics : bool
            是否需要绘制图片，默认为 False
        statistics : str
            统计量，目前支持 "psi"
    Returns:
        res_df: DataFrame
            两人群分布与 psi
        psi: float64
            psi
    Notes:
        对于 psi，<0.1 说明两人群分布差异很小，0.1~0.25 差异一般，≥0.25 差异很大
    """
    def is_covered(feature):
        if type(feature) == np.dtype('bool'):
            return feature not in col_no_cover_dict['bool']
        if type(feature) == np.dtype('int64'):
            return feature not in col_no_cover_dict['int64']
        if type(feature) == np.dtype('float64'):
            return feature not in col_no_cover_dict['float64']
        if type(feature) == np.dtype('str'):
            return feature not in col_no_cover_dict['str']
        return feature not in col_no_cover_dict['object']
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [0, -1], 'float64': [-1.0], 'str': [
            "-1", "unknown"], 'object': ["-1", "unknown"], 'bool': []}

    def calpsi(a, e):
        e = max(e, 0.0005)
        a = max(a, 0.0005)
        return (a - e) * math.log(a / e)

    df = pd.DataFrame(list(zip(group_df_col, feature_df_col)))
    df.columns = ['group', 'feature']

    # 分别计算两人群中，各类型比例
    portion0, portion1 = {}, {}
    groups = df.groupby('group')
    g0 = groups.get_group(0)
    g1 = groups.get_group(1)
    tot0, tot1 = 0, 0
    groups = g0.groupby('feature')
    for each_group in groups:
        type_ = each_group[1]['feature'].to_list()[0]
        if is_covered(type_):
            tot0 += each_group[1].shape[0]
    for each_group in groups:
        type_ = each_group[1]['feature'].to_list()[0]
        if is_covered(type_):
            portion0[type_] = 1.0 * each_group[1].shape[0] / tot0

    groups = g1.groupby('feature')
    for each_group in groups:
        type_ = each_group[1]['feature'].to_list()[0]
        if is_covered(type_):
            tot1 += each_group[1].shape[0]
    for each_group in groups:
        type_ = each_group[1]['feature'].to_list()[0]
        if is_covered(type_):
            portion1[type_] = 1.0 * each_group[1].shape[0] / tot1

    # 绘制图像
    if draw_pics:
        portion_type_list = []
        for x in portion0:
            portion_type_list.append([portion0[x], x])
        portion_type_list = sorted(portion_type_list, reverse=True)
        portion_list, type_list = [], []
        for x in portion_type_list:
            portion_list.append(x[0])
            type_list.append(x[1])
        df_draw = pd.DataFrame(
            {"portion %s" % group0: portion_list}, index=type_list)
        df_draw.plot.pie(y="portion %s" % group0, figsize=(5, 5))

        portion_type_list = []
        for x in portion1:
            portion_type_list.append([portion1[x], x])
        portion_type_list = sorted(portion_type_list, reverse=True)
        portion_list, type_list = [], []
        for x in portion_type_list:
            portion_list.append(x[0])
            type_list.append(x[1])
        df_draw = pd.DataFrame(
            {"portion %s" % group1: portion_list}, index=type_list)
        df_draw.plot.pie(y='portion %s' % group1, figsize=(5, 5))

    # psi 计算
    psi_sum = 0
    psi_type_list = []  # <psi, type>
    for x in portion0:
        if x not in portion1:
            portion1[x] = 0
    for x in portion1:
        if x not in portion0:
            portion0[x] = 0
    for x in portion0:
        psi_value = calpsi(portion0[x], portion1[x])
        psi_sum += psi_value
        psi_type_list.append([psi_value, x])
    psi_type_list = sorted(psi_type_list, reverse=True)
    type_, ratio0, ratio1, psi_list = [], [], [], []
    if statistics == "psi":
        for item in psi_type_list:
            x = item[1]
            type_.append(x)
            ratio0.append(portion0[x])
            ratio1.append(portion1[x])
            psi_list.append(item[0])
        res_df = pd.DataFrame(
            data={'type': type_, '%s 比例' % group0: ratio0,
                  '%s 比例' % group1: ratio1, 'psi': psi_list})
        return res_df, psi_sum

    for item in psi_type_list:
        x = item[1]
        type_.append(x)
        ratio0.append(portion0[x])
        ratio1.append(portion1[x])
    res_df = pd.DataFrame(
        data={'type': type_,
              '%s 比例' % group0: ratio0, '%s 比例' % group1: ratio1})
    print("暂未支持此统计量查询")
    return res_df


# test code
if __name__ == '__main__':
    print('test')
    # df = pd.read_csv("./basic.csv", header = 0)
    # analyse_cover_rate(df, "basic_4", "ins", "water",
    # feature_list = ['age', 'gender', 'work_city'])

    # df = pd.read_csv('prob.csv', header = 0)
    # print(analyse_hist(df['prob4'], 'ins',
    # 'water', df['proba_sports'], draw_pics = True))

    # df = pd.read_csv('car.csv', header = 0)
    # print(analyse_01(df["car4"], "ins",
    # "water", df["own_car"], "no car",
    # "own car",  draw_pics = True, statistics = "chi2"))

    # df = pd.read_csv('basic.csv', header = 0)
    # print(analyse_pie(df["basic_4"],
    #       "ins", "water", df["age"], draw_pics = True))
