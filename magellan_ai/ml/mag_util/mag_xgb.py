# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import pandas as pd


def get_xgboost_feature_imps(xgb_model_, importance_type,
                             is_sorted=False, filter_zero_imps=False):
    """Calculate the importance of features according to xgboost

    Parameters
    ----------
    xgb_model_ : XGBoostClassifier
        Instance of xgboost model.

    importance_type : {'gain', 'weight', 'cover',
                       'total_gain' or 'total_cover'}
        Feature importance type.

    is_sorted : bool, default=False
        Whether to sort in descending order for the importance of features

    filter_zero_imps : bool, default=False
        Whether showed the features whose importance was 0 in the results

    Returns
    --------
    feature_imps_df_ : DataFrame
        Calculation results of feature importance

    Note
    -------
    get_xgboost_feature_imps is equivalent to integrating get_Score and feature_importances_.
    The get_xgboost_feature_imps can not only specify feature importance type
    , but also the importance of all features can be calculated. In addition, the importance
    of features can be sorted
    """

    feature_map_ = xgb_model_.get_booster(). \
        get_score(importance_type=importance_type)

    # Get the feature with zero importance
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
