import pandas as pd


# from xgboost import XGBClassifier


def get_xgboost_feature_imps(xgb_model_, importance_type,
                             is_sorted=False, filter_zero_imps=False):
    """根据XGBoost计算特征重要性
    :param:
    * xgb_model_: XGBoost 模型实例
    * importance_type: one of ["gain", "weight", "cover",
    *                  "total_gain" or "total_cover"]
    * is_sorted: False/True, sort feature_df_sorted_['importance'] by 'desc'.
    * filter_zero_imps: False/True, weather keep feature
    *                   which importance = 0.0 feature_imps_df_
    :return:
    * feature_imps_df_: pd.DataFrame() with column ['feature', 'importance']
    Note:
    * 该方法获取指定importance_type的数值
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


if __name__ == "__main__":
    # 测试代码
    print("---")
