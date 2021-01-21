# magellan_ai说明
这是一个旨在提供Machine Learning和Deep Learning标准计算工具的Python项目，定期发布到`pipy.org`。

目前提供以下模块：
* `dl`
  * `mag_dnn`
    * `mag_mining`: 提供了DeepFM进行数据挖掘的方法。
    * `mag_transform`: 提供了TF record 和其他数据类型转换的方法。
  * `mag_predict`
    * `mag_train`: 提供了通过keras训练模型的样例方法。
    * `mag_online`: 提供了模型上线用到的特征文件生成工具。
    * `magbernard_client`: 提供了Bernard预测服务客户端的连接方法。
  * `mag_nlp`: 提供了NLP相关的各种工具方法，包括分词，实体识别等功能。
* `ml`    
  * `mag_util`
    * `mag_metrics`: 提供了计算各种指标的工具方法，包括auc,ks,iv,psi等指标的计算。
    * `mag_xgb` : 提供了基于XGBoost模型计算特征重要度的方法。
    * `mag_uap` : 提供了两人群差异分析方法。
    * `mag_calibrate`: 提供了模型分数校准方法，包括保序回归校准，高斯校准以及得分校准。
  * `mag_case`: 提供ml下的测试样例。
  

# 在线安装
建议使用官方镜像，安装最新版本。

```
$ pip install --index-url https://pypi.org/simple/mega-ai
```


# 使用教程
以`magllan_ai.ml.mag_util.mag_metrics`模块为例，安装完成之后，可以使用以下方法导入使用

```
from magllan_ai.ml.mag_util import mag_metrics

mag_metrics.show_func()
mag_metrics.cal_auc()
mag_metrics.cal_psi()
```

# 打包发布

```
$ cd /path/to/magellan_ml
$ python setup.py sdist bdist_wheel
$ pip install twine
$ twine upload dist/*
```