import pandas as pd
import numpy as np
from mag_util import metrics

# 计算AUC
def case_metrics_auc(path, prob_col, label_col):
    df = pd.read_csv(path)
    auc = metrics.cal_auc(df.loc[:, prob_col], df.loc[:, label_col])
    print(auc)

# case_metrics_auc('', '', '')