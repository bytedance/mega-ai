import pandas as pd

def get_xgboost_feature_imps(xgb_model_, importance_type, is_sorted=False, filter_zero_imps=False):
    """Get xgboost feature importance from XGBoost Model
    params:
    * xgb_model_: XGBoost model instance
    * importance_type: one of ["gain", "weight", "cover", "total_gain" or "total_cover"] 
    * is_sorted: False/True, sort feature_df_sorted_['importance'] by 'desc'.
    * filter_zero_imps: False/True, weather keep feature which importance = 0.0 feature_imps_df_
    result:
    * feature_imps_df_: pd.DataFrame() with column ['feature', 'importance']
    Note:
    * This method get feature importance as type of importance_type
    """

    feature_map_ = xgb_model_.get_booster().get_score(importance_type=importance_type)

    # find the feature which  importance value equal 0
    if not filter_zero_imps:
        for feat in model.get_booster().feature_names:
            if feat not in feature_map_.keys():
                feature_map_[feat] = 0

    feature_imps_df_ = pd.DataFrame()
    feature_imps_df_['feature'] = feature_map_.keys()
    feature_imps_df_['importance'] = feature_map_.values()

    if is_sorted:
        feature_imps_df_ = feature_imps_df_.sort_values(by='importances', ascending=False)

    return feature_imps_df_