# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import six.moves as sm
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from magellan_ai.ml.mag_util import mag_metrics


EPS = 1e-7


def show_func():
    """ 可视化函数
    """
    print("+---------------------------------------+")
    print("|analyse methods                        |")
    print("+---------------------------------------+")
    print("|feature_coverage_in_diff_people        |")
    print("|single_enum_feat_eval_diff_people      |")
    print("|single_continuity_feat_eval_diff_people|")
    print("+---------------------------------------+")


def feature_coverage_in_diff_people(
        df_,
        group_col,
        group_dict={},
        col_no_cover_dict={},
        col_handler_dict={},
        cols_skip=[],
        is_sorted=True):
    """calculate feature coverage for 2 groups
    Parameters
    -----------
    df_ : DataFrame
        Input file.

    group_col : list, ndarray, Series or DataFrame
        The column about group.

    group_dict : dict, optional
        Alias of group's name，default is equal to itself.

    col_no_cover_dict : dict, optional
        A custom feature specifies a non overriding value for the data type.
        default = {'int64': [-1], 'float64': [-1.0],
                    str': ["-1", "unknown"],
                    'object': ["-1", "unknown"], 'bool': []}

    col_handler_dict : dict, optional
        Dictionary of no-covered features.

    cols_skip : List, optional
        Ignore feature names for which feature coverage is calculated.

    is_sorted : bool, optional
        Whether to sort feature coverage

    Returns
    ----------
    feat_coverage_df : DataFrame
        The coverage of features and the corresponding data types

    Example
    -----------
    >>> df = pd.read_csv("./car.csv", header=0)
    >>> print(feature_coverage_in_diff_people(df,"car4",group_dict={
        0:"ins", 1:"water"}))
          feature  coverage_0  coverage_1 feat_type
    0  Unnamed: 0    1.000000    1.000000     int64
    1     car_did    1.000000    1.000000     int64
    2        car1    1.000000    0.987156   float64
    3        car2    1.000000    1.000000     int64
    4        car3    1.000000    1.000000     int64
    5        car4    1.000000    1.000000     int64
    6        car5    0.190126    0.262093   float64
    7     own_car    0.190126    0.262093   float64


    Notes
    -------
    There must be two valid tags in column group_df_col
    """
    # Default non overlay label
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}
    # Take out the characteristic information of the two groups
    groups = df_.groupby(group_col)
    group_dfs, indexs = [], []
    for index, group_df in groups:
        if index in col_no_cover_dict[str(df_[group_col].dtype)]:
            continue
        indexs.append(index)
        group_dfs.append(group_df)
    # If the number of population types is not 2, throw an exception
    try:
        if len(group_dfs) != 2:
            raise Exception("人群种类数不为 2")
    except Exception as err:
        print(err)
    # Generate datafraame
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

    # Modify two coverage column information
    df1.columns = ['feature', 'coverage_%s' % indexs[0], 'feat_type']
    df2.columns = ['feature', 'coverage_%s' % indexs[1], 'feat_type']
    del df1['feat_type']
    # Merge DataFrame
    res_df = pd.merge(df1, df2, how="inner", on="feature")

    return res_df


def single_enum_feat_eval_diff_people(
        group_df_col,
        feature_df_col,
        group_dict={},
        feature_dict={},
        col_no_cover_dict={},
        draw_pics=False):
    """
    1. The differences between the two groups in the characteristics of single
        enumeration class were analyzed.
        Examples of features: gender, occupation, favorite app
    2. Chi square test was performed when the characteristic values were 2,
        the PSI value was calculated when the characteristics were more than 2.
        Other statistical methods can be added later.

    Parameters:
    ------------
    group_df_col : Series
        The column about group.

    feature_df_col : Series
        The column in which the feature to be analyzed is located

    group_dict : dict, optional
        Alias of group's name，default is equal to itself.

    feature_dict : dict, optional
        Alias of feature's name，default is equal to itself.

    col_no_cover_dict : dict, optional
        dictionary of no-covered features.

    draw_pics : bool, optional
        Whether need to draw pictures, default is equal to false.

    Returns:
    -----------
    report : dict
        "DataFrame" : DataFrame
            Proportion of population features.
        "chi2" :float64
            when the number of features is equal to 2, calculate chi2-square.
        "psi" : float64
           If the feature number is greater than 2, calculate psi.
        "result" : str
            Conclusion of difference.

    Examples
    ------------
    >>> df = pd.read_csv('car.csv', header = 0)
    >>> dic = single_enum_feat_eval_diff_people(
        df['car4'], df['own_car'], group_dict={
            0: "ins", 1: "water"}, feature_dict={
            0: "not own car", 1: "own car"}, draw_pics=True)

          features  ins 人数    ins 比例  water 人数  water 比例
    0  not own car    6458  0.520471    102194  0.679617
    1      own car    5950  0.479529     48176  0.320383
    >>> print("chi2=%s, result=%s" % (dic["chi2"], dic["result"]))
        chi2=1308.0008370237344, result=根据卡方检验，两人群分布有明显差异


    >>> df = pd.read_csv("./t3.csv", header=0)
    >>> dic =  dic = single_enum_feat_eval_diff_people(
        df['t3_4'], df['career'], group_dict={
            0: "ins", 1: "water"}, draw_pics=True)
    >>> print(dic['DataFrame'])
                    feat_value       ins     water       psi
    0           gongwuyuan  0.036647  0.172391  0.210191
    1          blue_collar  0.794946  0.687720  0.015536
    2              courier  0.029653  0.013666  0.012385
    3                   it  0.022939  0.011836  0.007346
    4  individual_business  0.108635  0.106713  0.000034
    5              finance  0.007180  0.007674  0.000033
    >>> print("psi=%s, result=%s" % (dic["psi"], dic["result"]))
    psi=0.2455246396939325, result=有一定差异

    Notes:
    ----------
    1. group_df_col must have two valid tags, featute_ Df_ Col must have
        at least two valid tags,
        otherwise an exception is thrown.
    2. If the number of feature type is equal to 2, calculate chi square,
        else if the number of feature type is greater than 2, psi will
        be calculated
    3. If the number of feature type is greater than 25, the pie chart
        will not be drawn.
    4. In Chi square test, H0: π1 = π2, H1: π1 ≠ π2,
    5. Chi square test degree of freedom is 1, chi square value > = 3.841,
    p-value < = 0.05, which can be considered as significant difference.
    6. Psi related information:
        < 0.1: the difference is small
        0.1 ~ 0.25: average difference
        >= 0.25: very different
    """
    # default no_cover features
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}

    # merge group_df_col, featutre_df_col, delete rows
    # where no-cover feature locate at
    group_list, feature_list = [], []
    for group, feature in zip(group_df_col, feature_df_col):
        # Keep the information that the group and feature are valid
        # at the same time
        if group not in col_no_cover_dict[str(
                group_df_col.dtype)] and feature not in col_no_cover_dict[
                    str(feature_df_col.dtype)]:
            group_list.append(group)
            feature_list.append(feature)
            # Set the tag alias of the absent dict to itself
            if group not in group_dict:
                group_dict[group] = group
            if feature not in feature_dict:
                feature_dict[feature] = feature

    df = pd.DataFrame({'group': group_list, 'feature': feature_list})

    # Population type, feature type
    group_counter = len(df.groupby("group"))
    feature_counter = len(df.groupby('feature'))

    # Exception handling: when the number of characteristic types is
    # less than 2 and the number of population types is not equal to 2,
    #  an exception is thrown
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

    # The number and proportion of features in each population were calculated
    counter, ratio = {}, {}
    # Mapping of group tags to their feature information
    for group_tag in group_dict:
        tmp_counter, tmp_ratio = {}, {}
        # The mapping of feature tags to their frequency ratio in group
        # calculate frequency of occurrence about feature_tag in group_tag
        total = df.groupby("group").get_group(
            group_tag).shape[0]
        for feature_tag in feature_dict:
            cnt = df.groupby("group").get_group(group_tag).groupby(
                "feature").get_group(feature_tag).shape[0]
            tmp_counter[feature_tag] = cnt
            tmp_ratio[feature_tag] = 1.0 * cnt / total
        counter[group_tag] = tmp_counter
        ratio[group_tag] = tmp_ratio

    if feature_counter == 2:
        # chi square test
        # generate dataFrame，Frequency and proportion of two kinds of tags
        #  in two groups
        group_tag0, group_tag1 = group_dict.keys()  # group tags
        group0, group1 = group_dict.values()  # alias of group tags
        feature0, feature1 = feature_dict.values()
        # Alias for each feature tag
        cnt0, cnt1 = counter[group_tag0], counter[group_tag1]
        # In both groups, each feature: how many times does it appear
        ratio0, ratio1 = ratio[group_tag0], ratio[group_tag1]
        # In both groups, each characteristic: intra group share
        res_df = pd.DataFrame({'features': [feature0, feature1],
                               "%s 人数" % group0: list(cnt0.values()),
                               "%s 比例" % group0: list(ratio0.values()),
                               "%s 人数" % group1: list(cnt1.values()),
                               "%s 比例" % group1: list(ratio1.values())})
        # calculate chi-square
        a, b, c, d = list(cnt0.values()) + list(cnt1.values())
        n = a + b + c + d
        chi2 = 1.0 * n * (a * d - b * c) * (a * d - b * c) / \
            (a + b) / (a + c) / (b + d) / (c + d)
        # draw barh
        if draw_pics:
            index = [group0, group1]
            df_barh1 = pd.DataFrame({feature0: [cnt0[0], cnt1[0]], feature1: [
                                    cnt0[1], cnt1[1]]}, index=index)
            df_barh1.plot.barh(stacked=True)
            df_barh2 = pd.DataFrame(
                {feature0: [ratio0[0], ratio1[0]], feature1: [
                                    ratio0[1], ratio1[1]]}, index=index)
            df_barh2.plot.barh(stacked=True)
        # generate report
        report = {"DataFrame": res_df, "chi2": chi2}
        report["result"] = "根据卡方检验，两人群分布无明显差异" \
            if chi2 <= 3.841 else "根据卡方检验，两人群分布有明显差异"
        return report
    else:
        # calculate psi
        def calpsi(a, e):
            e = max(e, 0.0005)
            a = max(a, 0.0005)
            return (a - e) * np.log(a / e)
        # Frequency of various features in two populations
        portion0, portion1 = ratio.values()
        group0, group1 = group_dict.values()  # Group tag alias

        # If the key in the grouping of portion0 does not fall into portion1,
        # then the corresponding value of portion1 is 0
        for x in portion0:
            if x not in portion1:
                portion1[x] = 0

        # If the key in the grouping of portion1 does not fall into portion0,
        # then the corresponding value of portion0 is 0
        for x in portion1:
            if x not in portion0:
                portion0[x] = 0

        # The two groups of people were transformed into data frames and
        #  stitched according to the eigenvalues
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

        # calculate psi
        psi_li = []  # psi List
        psi_sum = 0
        for i in sm.range(len(res_df)):
            psi_value = calpsi(res_df[group0][i], res_df[group1][i])
            psi_sum += psi_value
            psi_li.append(psi_value)

        # The data frame showing the proportion of tags
        # in the population is generated,
        # and the feature tags are sorted according to the difference size
        res_df["psi"] = psi_li
        res_df = res_df.sort_values(by=['psi'], ascending=False)
        res_df.index = [i for i in sm.range(res_df.shape[0])]

        # Draw a pie chart of two groups
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

        # generate report
        report = {"DataFrame": res_df, "psi": psi_sum}
        if psi_sum < 0.1:
            report["result"] = "差异很小"
        elif psi_sum < 0.25:
            report["result"] = "有一定差异"
        else:
            report["result"] = "有较大差异"

        return report


"""
1. Analyze the more continuous feature differences
between the two groups from the mean value (e.g. probability label)
2. Draw the hist of two groups
3. At present, t-test / Welch is supported, the effect is not good,
and the follow-up test method needs to be updated.
"""


def single_continuity_feat_eval_diff_people(
        group_df_col,
        feature_df_col,
        group_dict={},
        col_no_cover_dict={},
        draw_pics=False):
    """
    Parameters:
    -----------
    group_df_col : Series
        The column about group.
    feature_df_col : Series
        The column in which the feature to be analyzed is located
    group_dict : dict, optional
        Alias of group's name，default is equal to itself.
    col_no_cover_dict : dict, optional
        Dictionary of no-cover features
    draw_pics : bool, optional
        Whether need to draw pictures, default is equal to false.

    Returns:
    ------------
    report : dict
        "DataFrame" : DataFrame
            Mean variance of two populations
        "p-value" : float
            t-test / Welch's result
        "result" : str
            Difference conclusion

    Example
    ----------
    >>> df = pd.read_csv('prob.csv', header = 0)
    >>> dic = single_continuity_feat_eval_diff_people(
        df["group"], df["pregnant_proba"], group_dict={
            0: "ins", 1: "water"}, draw_pics=True)
      features       ins     water
    0     mean  0.277334  0.272714
    1      std  0.209411  0.168556
    p-value=0.001006306471087986, result=有较大差异

    Notes:
    -------------
    The mean values of H0 and H1 are the same,
    but the mean values of H1 are different.
    This function selects t-test / Welch test according to
    the variance of two populations and the number of samples
    """
    # no-cover values
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}
    # merge group_df_col, featutre_df_col
    group_list, feature_list = [], []
    for group, feature in zip(group_df_col, feature_df_col):
        # be coverd at same time, in group and feature.
        if group not in col_no_cover_dict[str(
                group_df_col.dtype)] and feature > -EPS:
            group_list.append(group)
            feature_list.append(feature)
            if group not in group_dict:
                group_dict[group] = group

    group0, group1 = list(group_dict.values())
    df = pd.DataFrame({'group': group_list, 'feature': feature_list})

    # group counter
    group_counter = len(df.groupby("group"))
    # If group_counter = 2, throw exception
    try:
        if group_counter != 2:
            raise Exception("人群种类数不为 2")
    except Exception as err:
        print(err)

    # If feature is equal to obj，throw exception
    try:
        if str(df['feature'].dtype) == "object":
            raise Exception("fearute 为 object")
    except Exception as err:
        print(err)

    # The mean value and variance of the two populations were calculated,
    # and the dataframe was generated
    groups = df.groupby('group')
    vec0 = groups.get_group(0)['feature']
    vec1 = groups.get_group(1)['feature']
    features = ['mean', 'std']
    info0 = [vec0.mean(), vec0.std()]
    # The mean and variance of the first group were calculated
    info1 = [vec1.mean(), vec1.std()]
    # The mean and variance of the second group were calculated
    res_df = pd.DataFrame(
        data={'features': features, group0: info0, group1: info1})

    # Draw distribution histogram
    if draw_pics:
        plt.hist(vec0, bins=100, color="#FF0000", alpha=.9), plt.title(
            "%s" % group0), plt.show()
        plt.hist(vec1, bins=100, color="#FF0000", alpha=.9), plt.title(
            "%s" % group1), plt.show()

    # t test / Welch test
    L = stats.ttest_ind(vec0, vec1, equal_var=True if stats.levene(
        vec0, vec1).pvalue <= 0.05 else False)

    # generte reports
    report = {"DataFrame": res_df, "p-value": L.pvalue}
    report["result"] = "有较大差异" if L.pvalue < 0.05 else "无明显差异"
    return report
