# magellan_ml说明
这是一个旨在提供Machine Learning和Deep Learning标准计算工具的Python项目，定期发布到`pipy.org`。

目前提供以下模块：
* `mag_util`
  * `metrics`: 提供了计算各种指标的工具方法，包括auc,ks,iv,psi等指标的计算。
* `mag_nlp`: 提供了NLP相关的各种工具方法，包括分词，实体识别等功能。
* `mag_case`: 提供各种教程。

# 在线安装
建议使用官方镜像，安装最新版本。

```python
pip install magellan-ml -i https://pypi.Python.org/simple/
```

# 本地安装
如果要给该项目贡献代码，可以在本地调试好后测试，本地安装方法

```python
$ git clone git@code.byted.org:cfalg/magellan_ml.git
$ cd magellan_ml
$ python install .
```

# 使用教程
以`mag_util.metrics`模块为例，安装完成之后，可以使用以下方法导入使用

```
from mag_util import mag_metrics

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