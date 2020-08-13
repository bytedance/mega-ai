#coding=utf-8
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from scipy.stats import kstest
import numpy as np
import math

EPS = 1e-7

# 1. 分析两人群在单个枚举类特征上差异。特征的例子：性别，职业，最喜欢的 app
# 2. 特征取值为 2 种时，进行卡方检验，特征大于 2 种时，计算 psi 值。其它统计方法后续可以添加。
def single_enum_feat_eval_diff_people(group_df_col, feature_df_col, group_dict = {}, feature_dict = {}, col_no_cover_dict={}, draw_pics=False):
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
    ## 默认非覆盖标签
    if not col_no_cover_dict:
        col_no_cover_dict = {'int64': [-1], 'float64': [-1.0],
                             'str': ["-1", "unknown"],
                             'object': ["-1", "unknown"], 'bool': []}

    # 删掉无意义信息
    group_list, feature_list = [], []
    for group, feature in zip(group_df_col, feature_df_col):
        # 保留 group 和 feature 同时有效的信息
        if group not in col_no_cover_dict[str(group_df_col.dtype)] and feature not in col_no_cover_dict[str(feature_df_col.dtype)]:
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
    
    if feature_counter <= 1:
        print("此标签特征种类小于 2，无法差异分析")
        return None
    
    if group_counter != 2:
        print("人群种类数不为 2");
        return None

    # 计算各人群中各 feature 个数与比例
    counter, ratio = {}, {} # 人群标签向其特征信息的映射
    for group_tag in group_dict:
        #print(group_dict[group_tag])
        tmp_counter, tmp_ratio = {}, {} # group_tag 人群中，特征标签向其出现次数，比例的映射
        # 计算 group_tag 人群中 feature_tag 出现次数
        total = df.groupby("group").get_group(group_tag).shape[0]
        for feature_tag in feature_dict:
            cnt = df.groupby("group").get_group(group_tag).groupby("feature").get_group(feature_tag).shape[0]
            tmp_counter[feature_tag] = cnt
            tmp_ratio[feature_tag] = 1.0*cnt/total
        counter[group_tag] = tmp_counter
        ratio[group_tag] = tmp_ratio

    if feature_counter == 2:
        # 卡方检验
        # 生成 dataFrame，两人群中两种标签出现次数与占比
        group_tag0, group_tag1 = group_dict.keys() # 各 group 标签 
        group0, group1 = group_dict.values() # 各 group 标签别名
        feature0, feature1 = feature_dict.values() # 每种 feature 标签的别名
        cnt0, cnt1 = counter[group_tag0], counter[group_tag1] # 两个组中，每个特征:出现几次
        ratio0, ratio1 = ratio[group_tag0], ratio[group_tag1] # 两个组中，每个特征:组内占比       
        res_df = pd.DataFrame({'features': [feature0, feature1], 
                                "%s 人数" % group0: list(cnt0.values()),
                                "%s 比例" % group0: list(ratio0.values()),
                                "%s 人数" % group1: list(cnt1.values()),
                                "%s 比例" % group1: list(ratio1.values())})
        # 卡方计算
        a, b,c,d = list(cnt0.values()) + list(cnt1.values())
        n = a + b + c + d
        chi2 = 1.0 * n * (a*d-b*c) * (a*d-b*c) / (a+b) / (a+c) / (b+d) / (c+d)
        # 绘制 barh
        if draw_pics:
            index = [group0, group1]
            df_barh1 = pd.DataFrame({feature0: [cnt0[0], cnt1[0]], feature1: [cnt0[1], cnt1[1]]}, index = index)
            df_barh1.plot.barh(stacked=True)
            df_barh2 = pd.DataFrame({feature0: [ratio0[0], ratio1[0]], feature1: [ratio0[1], ratio1[1]]}, index = index)
            df_barh2.plot.barh(stacked=True)
        # 生成报告            
        report = {"DataFrame":res_df, "chi2":chi2}  
        report["result"] = "根据卡方检验，两人群分布无明显差异" if chi2 <= 3.841 else "根据卡方检验，两人群分布有明显差异"
        return report      
    else:
        # psi 计算
        def calpsi(a, e):
            e = max(e, 0.0005)
            a = max(a, 0.0005)
            return (a - e) * math.log(a / e)

        portion0, portion1 = ratio.values() # 两个人群各种特征出现频率
        group0, group1 = group_dict.values() # 各 group 标签别名
        
        # 如果portion0的分组中key没有落入portion1，那么对应portion1的取值为0
        for x in portion0:
            if x not in portion1:
                portion1[x] = 0

        # 如果portion0的分组中key没有落入portion1，那么对应portion1的取值为0
        for x in portion1:
            if x not in portion0:
                portion0[x] = 0

        # 分别将两组人群变成数据框并根据特征值进行拼接
        group0_df = pd.DataFrame.from_dict(portion0, orient="index", columns=[group0]).reset_index().rename(columns={"index": "feat_value"})
        group1_df = pd.DataFrame.from_dict(portion1, orient="index", columns=[group1]).reset_index().rename(columns={"index": "feat_value"})
        res_df = pd.merge(group0_df, group1_df, how="inner", on="feat_value")

        # psi 计算
        psi_li = [] # psi List
        psi_sum = 0
        for i in range(len(res_df)):
            psi_value = calpsi(res_df[group0][i], res_df[group1][i])
            psi_sum += psi_value
            psi_li.append(psi_value)

        # 生成显示标签在人群中占比的 DataFrame，并按差异大小，对特征标签排序
        res_df["psi"] = psi_li
        res_df = res_df.sort_values(by = ['psi'],  ascending = False)
        res_df.index = [i for i in range(res_df.shape[0])] 

        # 绘制两人群饼图
        if draw_pics == True:
            df0_draw = pd.DataFrame(
            {group0: list(portion0.values())}, index=list(portion0.keys()))
            _ = df0_draw.plot.pie(y=group0, figsize=(5, 5))

            df1_draw = pd.DataFrame(
            {group1: list(portion1.values())}, index=list(portion1.keys()))
            _ = df1_draw.plot.pie(y=group1, figsize=(5, 5))

        # 生成报告
        report = {"DataFrame": res_df, "psi sum":psi_sum}
        if psi_sum < 0.1:
            report["result"] = "差异很小"
        elif psi_sum < 0.25:
            report["result"] = "有一定差异"
        else:
            report["result"] = "有较大差异"
        
        return report

def single_number_feat_eval_diff_people(group_df_col, feature_df_col, group_dict = {}, col_no_cover_dict={}, draw_pics = False):
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
    # 删掉无意义信息
    group_list, feature_list = [], []
    for group, feature in zip(group_df_col, feature_df_col):
        # 保留 group 和 feature 同时有效的信息
        if group not in col_no_cover_dict[str(group_df_col.dtype)] and feature > -EPS:
            group_list.append(group)
            feature_list.append(feature) 
            if group not in group_dict:
                group_dict[group] = group
    
    group0, group1 = list(group_dict.values())
    df = pd.DataFrame({'group':group_list, 'feature':feature_list})

    # 计算两个人群包均值与方差，并生成 DataFrame
    groups = df.groupby('group')
    vec0 = groups.get_group(0);
    vec1 = groups.get_group(1); 
    features = ['mean', 'std']
    info0 = [vec0.mean(), vec0.std()] # 计算第一组均值、方差
    info1 = [vec1.mean(), vec1.std()] # 计算第二组均值、方差
    res_df = pd.DataFrame(data = {'features': features, group0: info0, group1: info1}) 
    
    # 绘制分布图像
    if draw_pics == True:
        plt.hist(vec0, bins = 100, color="#FF0000", alpha=.9), plt.title("%s" % group0), plt.show()
        plt.hist(vec1, bins = 100, color="#FF0000", alpha=.9), plt.title("%s" % group1), plt.show()
    
    # t-test
    L = stats.ttest_ind(vec0, vec1, equal_var = True if stats.levene(vec0, vec1).pvalue <= 0.05 else False)

    # 生成结论报告
    report = {"DataFrame": res_df, "p-value": L.pvalue}
    report["result"] = "有较大差异" if L.pvalue < 0.05 else "无明显差异"
    return report

"""
df = pd.read_csv("./car.csv", header = 0)
dic = single_enum_feat_eval_diff_people(df['car4'], df['own_car'], group_dict = {0: "ins", 1: "water"}, feature_dict = {0:"not own car", 1:"own car"}, draw_pics = True)
print(dic['DataFrame'])
print("chi2=%s, result=%s"%(dic["chi2"],dic["result"]))
"""

df = pd.read_csv("./t3.csv", header = 0)
dic = single_enum_feat_eval_diff_people(df['t3_4'], df['career'], group_dict = {0: "ins", 1: "water"}, draw_pics = True)
print(dic['DataFrame'])
print("psi=%s, result=%s"%(dic["psi sum"],dic["result"]))

"""
df = pd.read_csv('./prob.csv', header = 0)
dic = single_number_feat_eval_diff_people(df["prob4"], df["proba_parenting"], group_dict = {0:"ins",1:"water"}, draw_pics=True)
print(dic['DataFrame'])
print("p-value=%s, result=%s"%(dic["p-value"],dic["result"]))
"""