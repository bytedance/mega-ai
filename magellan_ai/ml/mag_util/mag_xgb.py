# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import pandas as pd


def get_xgboost_feature_imps(xgb_model_, importance_type,
                             is_sorted=False, filter_zero_imps=False):
    """根据XGBoost计算特征重要性

    Parameters
    ----------
    xgb_model_ : XGBoostClassifier
                 XGBoost的模型实例.

    importance_type : {'gain', 'weight', 'cover',
                       'total_gain' or 'total_cover'}
                      目前提供的特征重要性准则有增益，权重，覆盖，总增益，总覆盖
    is_sorted : bool, default=False
                是否针对特征重要性进行倒序排列

    filter_zero_imps : bool, default=False
                       是否在结果中展示特征重要性为0的特征

    Returns
    --------
    feature_imps_df_ : DataFrame
                       特征重要性的计算结果


    Note
    -------
    get_xgboost_feature_imps 相当于综合了 get_score 以及 feature_importances_
    二者的优点， 不仅可以设置指定的特征重要度类型，而且可以计算所有特征的重要性，除此之外 也可以针对特征重要度进行排序
    """

    feature_map_ = xgb_model_.get_booster(). \
        get_score(importance_type=importance_type)

    # 获取特征重要度为零的特征
    if not filter_zero_imps:
        for feat in xgb_model_.get_booster().feature_names:
            if feat not in feature_map_.keys():
                feature_map_[feat] = 0

    feature_imps_df_ = pd.DataFrame()
    feature_imps_df_['feature'] = feature_map_.keys()
    feature_imps_df_['importance'] = feature_map_.values()
    feature_imps_df_ = feature_imps_df_.set_index("feature").loc[
                       xgb_model_.get_booster().feature_names,
                       :].reset_index().rename(columns={"index": "feature"})

    if is_sorted:
        feature_imps_df_ = feature_imps_df_.sort_values(
            by='importance', ascending=False)

    return feature_imps_df_
