import pandas as pd

from magellan_ai.magellan_ml.mag_util import mag_metrics


# 计算AUC
def case_metrics_auc(path, prob_col, label_col):
    df = pd.read_csv(path)
    auc = mag_metrics.cal_auc(df.loc[:, prob_col], df.loc[:, label_col])
    print(auc)
