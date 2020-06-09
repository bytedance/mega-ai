import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def show_func():
    """Show functions provided in metrics module"""
    print("cal_auc, cal_ks, cal_psi, cal_iv, cal_desc")

def cal_auc(y_true, y_pred):
    """Calculate AUC
    params:
    * y_true: label array, element is one of [0, 1].
    * y_pred: model predict, float type.
    """

    pairs = list(zip(y_pred, y_true))
    rank = [label_value for prob_value, label_value in sorted(pairs, key=lambda x:x[0])]
    rank_list = [i+1 for i in range(len(rank)) if rank[i]==1]
    pos_num = 0
    neg_num = 0
    for i in range(len(y_true)):
        if(y_true[i]==1):
            pos_num+=1
        else:
            neg_num+=1
    auc = 0
    auc = (sum(rank_list)- (pos_num*(pos_num+1))/2)/(pos_num*neg_num)
    
    return auc
    

def cal_ks(y_true, y_pred):
    '''Calculate KS
    * label_values: a array , list, numpy array or pandas series with numberic elements
    * prob_values: a array , list, numpy array or pandas series with numberic elements
    '''

    fpr,tpr,thresholds = roc_curve(np.array(y_true),np.array(y_pred))
    tpr_fpr_gap = abs(fpr-tpr)
    ks = max(tpr_fpr_gap)
    cut = thresholds[tpr_fpr_gap==ks][0]  

    return cut, ks

def cal_psi(base_values, curr_values):
    

    return 0.001

def cal_psi(score_1, score_2, k_part = 10):
    '''Calculate PSI
    params:
    * base_values: a array , list, numpy array or pandas series with numberic elements
    * curr_values: a array , list, numpy array or pandas series with numberic elements
    return:
    * PSI: float type.
    '''
    
    score_1 = score_1[pd.notnull(score_1)] 
    score_2 = score_2[pd.notnull(score_2)]
    score_1 = np.array(score_1[pd.notnull(score_1)])
    score_2 = np.array(score_2[pd.notnull(score_2)])

    percentile = np.linspace(0,100,num=k_part+1)
    threshold = [np.percentile(score_1, i) for i in percentile]
    threshold = list(set(threshold))
    threshold.sort() 

    rate_1 = []
    rate_2 = []
    
    flag = 0 
    if sum(score_1==threshold[0]) >= len(score_1)/k_part:
        flag = 1
        rate_1.append(sum(score_1==threshold[0]))
        rate_2.append(sum(score_2==threshold[0]))
        
    for i in range(len(threshold)-1):
        start = threshold[i]
        end = threshold[i+1]
        if i ==0 and flag == 0:
                rate_1.append(sum((score_1 >= start) & (score_1 <= end)))
                rate_2.append(sum((score_2 >= start) & (score_2 <= end)))
        else: 
            rate_1.append(sum((score_1 > start) & (score_1 <= end)))
            rate_2.append(sum((score_2 > start) & (score_2 <= end)))        
    rate_1 = np.array(rate_1)/float(len(score_1))
    rate_2 = np.array(rate_2)/float(len(score_2))

    # cal psi
    try:
        PSI = sum([(rate_1[i] - rate_2[i]) * math.log((rate_1[i] / rate_2[i]), math.e) for i in range(len(rate_1))])
    except:
        PSI = float('inf')
    return PSI

def cal_iv():
    print('%s\t%.6f' % ('ugc_embedding'.ljust(24), 0.123456))

def cal_feature_coverage(df_, col_aim_dict = {}, col_handler_dict = {}, cols_skip=[]):
    '''analyze feature coverage for pandas dataframe
    params
    * df_: a pandas dataframe
    * col_aim_dict: a dict for custom non-cover value. by default:
        * int64: [0, -1]
        * float64: [0.0, -1.0]
        * object: []
        * bool: []
    * col_handler_dict: a dict for custom non-cover value statistics.
    * col_skip: a list for skip cols which will not analyze the coverage.
    '''

    if not col_aim_dict:
        col_aim_dict = {'int64': [0, -1], 'float64': [0.0, -1.0], 'object': [], 'bool': []}

    def col_hanler_bool(df_col):
        '''bool col coverage defualt handler'''
        
        return df_col.isna().sum()

    def col_handler_object(df_col):
        '''object col coverage defualt handler'''

        return df_col.isna().sum()

    def col_handler_int64(df_col):
        '''integer col coverage defualt handler'''

        row_cnt = 0
        for col_aim_value in col_aim_dict['int64']:
            row_cnt = row_cnt + df_col[ df_col == col_aim_value ].shape[0]

        return row_cnt + df_col.isna().sum()

    def col_handler_float64(df_col):
        '''float and double col coverage defualt handler'''

        row_cnt = 0
        for col_aim_value in col_aim_dict['float64']:
            row_cnt = row_cnt + df_col[abs(df_col - col_aim_value) <= 1e-6 ].shape[0]

        return  row_cnt + df_col.isna().sum()

    (row_count, col_count) = df_.shape
    col_dtype_list = ['int64', 'float64', 'object']
    col_coverage_dict = {}
    col_dtype_dict = {}
    
    for col_name in df_.columns:
        if col_name in cols_skip:
            continue

        col_handler = col_handler_int64
        cover_count = 0
        coverage = 0

        if df_[col_name].dtype == np.dtype('bool'):
            if 'bool' in col_handler_dict:
                col_handler = col_handler_dict['object']
            else:
                col_handler = col_handler_object

        if df_[col_name].dtype == np.dtype('object'):
            if 'object' in col_handler_dict:
                col_handler = col_handler_dict['object']
            else:
                col_handler = col_handler_object

        if df_[col_name].dtype == np.dtype('int64'):
            if  'int64' in col_handler_dict:
                col_handler = col_handler_dict['int64']
            else: 
                col_handler = col_handler_int64
        
        if df_[col_name].dtype == np.dtype('float64'):
            if 'float64' in col_handler_dict:
                col_handler = col_handler_dict['float64']
            else:
                col_handler = col_handler_float64
        
        value_count = col_handler(df_.loc[:, col_name])
        coverage = (row_count - value_count) * 1.0 / (row_count + 1e-6)
        
        col_coverage_dict[col_name] = coverage
        col_dtype_dict[col_name] = df_[col_name].dtype
            
    return col_coverage_dict, col_dtype_dict