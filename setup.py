# coding: utf-8

import io
import os

from setuptools import find_packages, setup, Extension

# 包元信息
NAME = 'magellan-ai'
DESCRIPTION = 'Some computing tools with machine learning, python 3.6+.'
URL = 'https://code.byted.org/scm/python-boilerplate'  # TODO: 修改为项目实际仓库 URL
EMAIL = 'zhangliang.thanks@bytedance.com'
AUTHOR = 'zhangliang'

# 项目运行需要的依赖
REQUIRES = [
    'bytedbackgrounds>=0.0.2,<1.0.0',
    'bytedenv>=0.3.1,<1.0.0',
    'bytedlogger>=0.2.2,<1.0.0',
    'bytedlogid>=0.0.2,<1.0.0',
    'bytedmetrics>=0.6.1,<1.0.0',
    'bytedservicediscovery~=0.11',
    'gunicorn>=19,<21',
    'ipaddress>=1.0.10,<2.0.0; python_version<"3.3"',
    'psutil>=5.6.0,<=6.0.0',
    'requests>=2.19.1,<3.0.0',
    'six>=1.13.0,<2.0.0',
    'thriftpy2>=0.4.7,<=1.0.0',
    'typing>=3.6.4; python_version<"3.5"',
]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except IOError:
    long_description = DESCRIPTION

setup(
      name='byted'+NAME,
      version='1.0.0',
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      packages=find_packages(),
      install_requires=REQUIRES,
      python_requires='>=3.6',
      include_package_data=True
)
