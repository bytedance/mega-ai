import pandas as pd

from magellan_ai.magellan_ai.ml.mag_util import mag_metrics, mag_uap


# 计算AUC
def case_metrics_auc(path, prob_col, label_col):
    df = pd.read_csv(path)
    auc = mag_metrics.cal_auc(df.loc[:, prob_col], df.loc[:, label_col])
    print(auc)


if __name__ == "__main__":

    # 测试代码
    mag_metrics.show_func()
    data_df = pd.read_csv("/Users/bytedance/ByteCode"
                          "/magellan_ai/data/cs-training.csv", index_col=0)

    print(data_df.columns)
    print(mag_metrics.cal_iv(data_df, "SeriousDlqin2yrs"))
    print(mag_uap.show_func())
