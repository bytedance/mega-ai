import numpy as np
import pandas as pd

def analyze_feature_coverage(df_, 
                            col_aim_dict = {'int64': [0, -1], 'float64': [0.0, -1.0], 'object': [], 'bool': []}, 
                            col_handler_dict = {}, 
                            cols_skip=[]):
    '''analyze feature coverage for pandas dataframe
    * input
        * df_: a pandas dataframe
        * col_aim_dict: a dict for custom non-cover value. by default:
            * int64: [0, -1]
            * float64: [0.0, -1.0]
            * object: []
            * bool: []
        * col_handler_dict: a dict for custom non-cover value statistics.
        * col_skip: a list for skip cols which will not analyze the coverage.
    '''

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