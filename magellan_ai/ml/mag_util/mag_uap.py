#coding=utf-8
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from scipy.stats import kstest
import numpy as np
import math

################################################################################
BAN_WORD = "banword_dAc2D#$!SAD@2exdevcsS312SZ!ds1xswxxdceccsdabhuve3"

def analyse_cover_rate(df, tag_col, tag_name = BAN_WORD, print_logs = False):
    """Calculate cover_rate
    :params df: dataframe
    :params tag_col: 标签在 DF 中哪一列
    :params tag_name: 标签的含义，不填则为 tag_col
    :params print_logs: 是否打印 Log
    :return: cover_cnt 覆盖条数, tot 总共有几条, ratio 比率
    """
    if tag_name == BAN_WORD: 
        tag_name = tag_col
    groups = df.groupby(tag_col)
    tot = df.shape[0]
    null_group = groups.get_group("-1" if type(df[tag_col][1]) == (type)('cat') else -1)
    null_size = null_group.shape[0]
    cover_cnt = tot - null_size
    ratio = 1.0 * cover_cnt / tot
    if print_logs: 
        print("对于标签 %s, p_date 分区下有 %d 条，覆盖 %d 条，覆盖率为 %.5f" % (tag_name, tot, cover_cnt, 1.0 * cover_cnt / tot))
    return cover_cnt, tot, ratio

###############################################################################
def analyse_pie(df, group_col, group0, group1, tag_col, tag_name = BAN_WORD, d = 0.1, k_th_max = 3, print_logs = False, statistics = "psi"):
    """Analyse enums
    :params df: dataframe
    :params group_col: 表示人群的标签在哪一列。
    :params group0: 0 表示什么人群
    :params group1: 1 表示什么人群
    :params tag_col: 标签在 DF 中哪一列
    :params tag_name: 标签的名字，不填设为 tag_col
    :params d: 饼图偏离中心量
    :params k: 列举 psi 前 k 大值
    :params print_logs: 是否打印 Log
    :params statistics: 默认为 psi
    :return: psi 值
    """          
    if tag_name == BAN_WORD:
        tag_name = tag_col
    def ok(str):
        if str == -1 or str == "-1" or str == "-1.0":
            return 0
        if str == "" or str == "unknown":
            return 0
        return 1

    def calpsi(a, e):
        e = max(e, 0.0005)
        a = max(a, 0.0005)
        return (a - e) * math.log(a / e)

    def check_psi_sum(psi_sum):
        if print_logs:
            if psi_sum >= 0.2:
                print("psi 为 %.5g 差异显著" % psi_sum)
            elif psi_sum >= 0.1:
                print("psi 为 %.5g 有一定差异" % psi_sum)
            else:
                print("psi 为 %.5g 差异不明显" % psi_sum)

    def psi(dic1, dic2, k_th_max = 3):
        g0 = group0
        g1 = group1
        ## union
        for x in dic1:
            if x not in dic2:
                dic2[x] = 0

        for x in dic2:
            if x not in dic1:
                dic1[x] = 0

        sum_psi = 0
        diff = []

        for x in dic1:
            tmp = calpsi(dic1[x], dic2[x])
            sum_psi = sum_psi + tmp
            diff.append(list([- tmp, x]))

        diff.sort()
        check_psi_sum(sum_psi)
        if print_logs:
            print("其中差异最为显著的 %d 个标签为：" % min(k_th_max, len(diff)))
            for i in range(min(k_th_max, len(diff))):
                name = diff[i][1]
                # print(name, abs(diff[i][0]))
                print("%s, psi 为 %.8f, %s 人群的占比为 %.3g%c, %s 人群占比为 %.3g%c, %s 人群中比例更高" %(name, abs(diff[i][0]), g0, dic1[name] * 100, '%', g1, dic2[name] * 100, '%', (g0 if dic1[name] > dic2[name] else g1) ))
        return sum_psi
    
    def tackle(L, mp, d = 0.1): ## draw pie chart about list L, put result into mp(dic: name -> portion)
        # print(type(L))
        L = list(L)
        for item in L:
            if ok(item):
                mp[item] = 0

        for item in L:
            if ok(item):
                mp[item] = mp[item] + 1

        labels,sizes,explode,lis = [],[],[],[]
        tot = 0
        for x in mp:
            tot += mp[x]

        for x in mp:
            if mp[x] > 0:
                lis.append(x)

        lis = sorted(lis)
        for x in lis:
            if print_logs:
                print(x, mp[x])
            labels.append(x)
            sizes.append(mp[x])
            explode.append(d)
            d = d + 0.1

        if print_logs == True:
            if len(labels) <= 15:
                plt.pie(sizes,explode=explode,labels=labels,shadow=False,startangle=150)
                plt.show()  
        for x in mp:
            mp[x] = mp[x] / tot
        return
    
    g0 = group0
    g1 = group1
    group_col = group_col
    
    if print_logs:
        print("%s 人群和 %s 人群在 %s 上的差异分析" % (g0, g1, tag_name))
        print("======================================================")
    
    group_by_name=df.groupby(group_col) 
    ## g0
    ins_df_0 = group_by_name.get_group("0" if type(df[group_col][0]) == type("cat") else 0)
    # dd = ins_df_0.groupby(3)
    # print(dd.groups)
    if print_logs:
        print("%s 人群各类用户人数与饼图" % g0)
    mp1 = {}
    
    tackle(list(ins_df_0[tag_col]), mp1)
    if print_logs:
        print("------------------------------------------------------")
    ## g1
    ins_df_1 = group_by_name.get_group("1" if type(df[group_col][0]) == type("cat") else 1)
    if print_logs:
        print("%s 人群各类用户人数与饼图" % g1)
    mp2 = {}
    tackle(ins_df_1[tag_col], mp2)

    lis = []
    for x in mp1:
        if x not in mp2:
            mp2[x] = 0
    for x in mp2:
        if x not in mp1:
            mp1[x] = 0
    for x in mp1:
        if (ok(x)):
            lis.append(x)

    X, Y = [], [] 
    if print_logs:
        print("------------------------------------------------------")
        print("%s 人群和 %s 人群，比例对比" % (g0, g1))
        for x in lis:
            X.append(str(x) + "_%s(%.1f%c)" % (g0, mp1[x] * 100, '%'))
            X.append(str(x) + "_%s(%.1f%c)" % (g1, mp2[x] * 100, '%'))
            Y.append(mp1[x])
            Y.append(mp2[x])
            print("%s %.8f %.8f" % (x, mp1[x], mp2[x]))

        if (len(X) <= 15):
            plt.barh(X, Y)  
            plt.title('portion compare')
            plt.show() 

        print("------------------------------------------------------")
    return psi(mp1, mp2, k_th_max)
#####################################################################################

def analyse_01(df, group_col, group0, group1, tag_col, tag0, tag1, tag_name = BAN_WORD, print_logs = False, statistics = "chi2"): 
    """Analyse 01 tags
    :params df: dataframe
    :params group_col: 表示人群的标签在哪一列。
    :params group0: 0 表示什么人群
    :params group1: 1 表示什么人群
    :params tag_col: 标签在 DF 中哪一列
    :params tag0: 0 表示标签的含义
    :params tag1: 1 表示标签的含义
    :params tag_name: 标签的名字，不填设为 tag_col
    :params print_logs: 是否打印 Log
    :params statistics: 默认为卡方值
    :return: chi2 卡方值
    """    
    if tag_name == BAN_WORD:
        tag_name = tag_col
    
    def analyse_bool(Mat, groupOfX, groupOfY, not_xxx, xxx, tag_name):
        # Mat = [[0 for i in range(2)] for j in range(2)]
        ##  count
        ##   G1 G2
        ## N 
        ## Y  
        y1, y2 = Mat[0], Mat[1]
        if print_logs == True:
            print("   %s %s" % (not_xxx, xxx))
            print("%s %d %d" % (groupOfX, Mat[0][0], Mat[1][0]))
            print("%s %d %d" % (groupOfY, Mat[0][1], Mat[1][1]))
            labels = [groupOfX, groupOfY]
            plt.barh(labels, y1, color='green', label=not_xxx) ## Y_label need change
            plt.barh(labels, y2, left=y1, color='red', label=xxx) ## Y_label need change
            plt.title(tag_name)  
            plt.legend(loc=[1, 0])  
            plt.xlabel("Person")  
            plt.show()

        ## cal chi^2
        #n(ad-bc)^2/(a+b)(c+d)(a+c)(b+d)
        n = Mat[0][0] + Mat[0][1] + Mat[1][0] + Mat[1][1]
        a, b, c, d = Mat[0][0], Mat[0][1], Mat[1][0], Mat[1][1]
        chi2 = 1.0*n*(a*d-b*c)*(a*d-b*c)/(a+b)/(c+d)/(a+c)/(b+d)
    
        if print_logs == True:
            print("chi^2 = %.5g" % chi2)
            if chi2 >= 7.879:
                print("在 %s 上差异显著" % tag_name)
            else:
                print("在 %s 上，无显著差异" % title)

        ## percent
        C = [0, 0]
        C[0] = Mat[0][0] + Mat[1][0]
        C[1] = Mat[0][1] + Mat[1][1]
        if C[0] == 0 or C[1] == 0:
            print("ERROR, percent divede zero exception!")
            return
        for col in range(2):
            for row in range(2):
                Mat[row][col] /= (C[col] * 1.0) 
        y1, y2 = Mat[0], Mat[1]            
        if print_logs == True:
            print("  %s %s" % (not_xxx, xxx))
            print("%s %.3g %.3g" % (groupOfX, Mat[0][0], Mat[1][0]))
            print("%s %.3g %.3g" % (groupOfY, Mat[0][1], Mat[1][1]))
            labels = [groupOfX, groupOfY]
            plt.barh(labels, y1, color='green', label=not_xxx)
            plt.barh(labels, y2, left=y1, color='red', label=xxx)
            plt.title(tag_name) 
            plt.legend(loc=[1, 0])  
            plt.xlabel("Percent")  
            plt.show()
        return chi2

    if print_logs == True:
        print("%s 人群和 %s 人群在 %s 上差异分析" % (group0, group1,tag_name))

    gs = df.groupby(group_col)

    Mat = [[0 for i in range(2)] for j in range(2)]
    ## tag_0
    df0 = gs.get_group("0" if type(df[group_col][0]) == type("cat") else 0) 
    #print(df0)
    df0_ = df0.groupby(tag_col)
    Mat[0][0] = len(df0_.get_group(0))
    Mat[1][0] = len(df0_.get_group(1))
    #print(df0_.groups)

    # #tag_1
    df1 = gs.get_group("1" if type(df[group_col][0]) == type("cat") else 1)
    df1_ = df1.groupby(tag_col)
    Mat[0][1] = len(df1_.get_group(0))
    Mat[1][1] = len(df1_.get_group(1))

    #print(df1_.groups)
    return analyse_bool(Mat, group0, group1, tag0, tag1, tag_name)

##########################################################################################################################
def analyse_hist(df, group_col, group0, group1, tag_col, name = BAN_WORD, print_logs = False, statistics = "t-test"):
    """Analyse number
    :params df: dataframe
    :params group_col: 表示人群的标签在哪一列。
    :params group0: 0 表示什么人群
    :params group1: 1 表示什么人群
    :params tag_col: 标签在 DF 中哪一列
    :params name: 标签的名字，不填赋为 tag_col
    :params print_logs: 是否打印 Log
    :params statistics: 统计量，默认为 t-test
    :return: tag0 人群均值，tag1 人群均值，tag0 人群方差，tag1 人群方差，t-test 的 pvalue
    """          
    if name == BAN_WORD:
        name = tag_col
        
    g0 = group0
    g1 = group1
    group_col = group_col
    group_by_name=df.groupby(group_col) 

    ins_df_0 = group_by_name.get_group("0" if type(df[group_col][0]) == type("cat") else 0) 
    #print(cal(ins_df_0[col]))
    a = ins_df_0[tag_col]
    res = []
    for x in a:
        if x >= 0:
            res.append(x)
    a = res
    

    ins_df_1 = group_by_name.get_group("1" if type(df[group_col][0]) == type("cat") else 1)
    b = ins_df_1[tag_col]
    res = []
    for x in b:
        if x >= 0:
            res.append(x)
    b = res
              
    if print_logs == True:
        #print("%s 人群均值 %.5f，方差 %.5f" %(g0, np.mean(a), np.std(a))
        #print("%s 人群均值 %.5f，方差 %.5f" %(g1, np.mean(b), np.std(b))
        print("%s 人群分布" % g0)
        plt.hist(a, bins = 100, color="#FF0000", alpha=.9)
        plt.title('%s about %s' % (name, g0))
        plt.show()
    L = stats.ttest_ind(a, b, equal_var = True if stats.levene(a, b).pvalue <= 0.05 else False)

    u = np.mean(a)
    d = np.std(a)

    if print_logs == True:
        print('%s 人群均值 %.5f，方差 %.5f' %(g1, np.mean(a), np.std(a)))
        print("%s 人群分布" % g1)
        plt.hist(b, bins = 100, color="#FF0000", alpha=.9)
        plt.title('%s about %s' % (name, g1))
        plt.show()
        x = ['%s ave' % g0,'%s ave' % g1,'%s var' % g0,'%s var' % g1] 
        y = [np.mean(a), np.mean(b), np.std(a), np.std(b)]
        print(L)
    
        if L.pvalue <= 0.05:
            print("%s 人群和 %s 人群在 %s 上，有较大差异" % (g0, g1, name))
            if np.mean(a) < np.mean(b):
                print("%s 人群 %s 更低" % (g0, name))
            else:
                print("%s 人群 %s 更低" % (g1, name))
        else:
            print("%s 人群和 %s 人群在 %s 上，无较大差异" % (g0, g1, name))
        plt.bar(x, y)
        plt.title('average and variance of ' + name)
        plt.show()
        print('\n\n')
    return np.mean(a),np.mean(b),np.std(a),np.std(b),L.pvalue


if __name__ == '__main__':
    """ Test code
    df = pd.read_csv("prob.csv", header = 0)
    analyse_hist(df, "prob4", "ins", "water", "proba_health", print_logs = True);
    
    df = pd.read_csv("basic.csv", header = 0)
    analyse_pie(df, "basic_4", "ins", "water", "age", print_logs = True)
    
    df = pd.read_csv("other_tags.csv", header = 0)
    analyse_01(df, "4", 'ins', 'water', 'own_house', 'not own', 'own', print_logs = True)
    """
    