import pandas as pd

def get_xgboost_feature_imps(xgb_model_, ascending=False):
    """Get xgboost feature importance from XGBoost Model
    * xgb_model_: XGBoost model instance
    * ascending: False/True, sort feature_df_sorted_['importances'] by 'ascending'.
    Note:
    * This method get feature importance as type of importance_type
    """
    feature_df_ = pd.DataFrame()
    feature_df_sorted_ = feature_df_

    return feature_df_, feature_df_sorted_