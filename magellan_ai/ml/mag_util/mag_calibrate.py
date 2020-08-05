# # -*- coding: utf-8 -*
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression


"""
    Author: huangning.honey@bytedance.com
    Date: 2020/08/04
    Function: 实现不同的模型分数校准方法
"""


def isotonic_calibrate(input_df, fit_num=3, bin_num=1000, bin_type="same_frenquency",
                       bin_value_method="mean", is_poly=False):
    '''
        bin_type: 分箱方式，默认是等频分箱
        bin_value_method: 箱值计算方法："mean"
        is_poly: 是否针对分箱结果进行多项式拟合，进而于保序回归拟合结果进行比较
    '''
    # 如果数据量小于默认的分箱个数，那么直接报错
    if len(input_df)<bin_num:
        raise Exception("数据量小于分箱个数")

    # 计算实际需要的箱子个数，eg: 10个数放到6个箱子中，如果等频，需要10个箱子
    # 默认等频分箱
    quantile = np.linspace(0, 1, bin_num+1)
    interval = pd.qcut(len(input_df), q=quantile, precision=3, duplicates='drop')

    # 增加分箱区间列
    input_df["interval"] = interval
    
    # 计算每个箱内正样本的比例，如果是空箱，那么直接设置成1
    pos_rate = input_df.groupby(by="interval").apply(lambda x: sum(x['label']==1)/len(x['label'])
                                                 if len(x['label'])!=0 else 1 )

    # 重新计算分箱的个数（因为之前的分箱操作导致分箱数量减少）
    bin_num = len(pos_rate)
    # 二维坐标点，第1维是桶值，第2维是真实的通过率
    coordinate_li = []


    # 计算每个分箱的代表值
    for i in range(bin_num):
        temp_df = input_df[input_df["interval"] == posRate_series.index[i]]
        if bin_value_method == "mean":
            temp = [temp_df["proba"].mean(), posRate_series[i]]
        elif bucketType == "medium":
            temp = [temp_df["proba"].median(), posRate_series[i]]
        elif bucketType == "mode":
            # 众数可能存在多个，默认取多个众数中的第一个
            temp = [temp_df["proba"].mode()[0], posRate_series[i]]
        else:
            print("您输入的箱值不在程序的处理范围之内，请重新输入...")
            break
        coordinate_li.append(temp)

    # 将坐标点变成数据框，方便后面模型训练
    data_df = pd.DataFrame(coordinate_li, columns=["bin_value", "pos_rate"])

    # 向数据集中添加（0，0）和（1，1），相当于增加两个箱，使得保序模型的横坐标可测范围是从[0,1]
    data_df = Iso_df.append({"bin_val":0, "pos_rate":0}, ignore_index=True)
    data_df = Iso_df.append({"bin_val":1, "pos_rate":1}, ignore_index=True)

    # 为了后面的保序回归模型，需要将bin_val从小到大排序
    data_df.sort_values(by="bin_val", inplace=True)

    # 训练保序回归模型
    iso_reg = IsotonicRegression(increasing=True)
    iso_reg.fit(X=data_df.values[:, 0], y=data_df.values[:, 1])
    y_pred = iso_reg.predict(data_df.values[:, 0])
    data_df['iso_pred'] = y_pred

    fitNum = 3
    poly_li = []  # 保存每个多项式模型
    
     # 计算全量样本的保序回归预测概率
    y_pred = IRreg.predict(input_df["proba"])
    input_df["iso_pred"] = y_pred
    
    # 判断是否还有多项式需要拟合
    if is_poly:

        # 拟合额外的多项式
        if fitNum > 9 or fitNum < 0:
            print("抱歉，您输入的拟合次数超过阈值9或拟合次数为负数，请重新输入")
        elif fitNum > 0:
            for i in range(fitNum):
                z = np.polyfit(data_df['bin_value'], data_df['pos_rate'], i+1)
                p = np.poly1d(z)
                poly_li.append(p)
                yfit = p(data_df['bin_value'])
                colname = "{}_polynomial".format(i+1)
                data_df[colname] = yfit

        # 针对全量数据增加多项式的预测结果
        for index, polycol in enumerate(poly_li):
            polyname = "{}_polynomial".format(index+1)
            polyVal = polycol(input_df["proba"])
            input_df[polyname] = polyVal

    return input_df

if __name__ == "__main__":

    test_df = pd.DataFrame({"label": [1, 0, 0, 1, 0],
                            "prob": [0.7, 0.3, 0.2, 0.3, 0.6]})
    print("测试数据框")
    print(test_df)
    # res = isotonic_calibrate(test_df, bin_num=3)
    # print(res)
