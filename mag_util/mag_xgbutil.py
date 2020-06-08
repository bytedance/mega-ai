import pandas as pd

def get_xgboost_feature_imps(xgb_model_, importance_type='total_gain', is_sorted=False, filter_zero_imps=False):
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

    feature_imps_df_ = pd.DataFrame()

    return feature_imps_df_