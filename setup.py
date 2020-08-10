# coding: utf-8
from setuptools import find_packages, setup

import io
import os

# 包元信息
NAME = 'magellan-ai'
DESCRIPTION = 'Some computing tools with machine learning, python 3.6+.'
URL = 'https://code.byted.org/scm/python-boilerplate'  # TODO: 修改为项目实际仓库 URL
EMAIL = 'zhangliang.thanks@bytedance.com'
AUTHOR = 'zhangliang'

# 项目运行需要的依赖
REQUIRES = ["numpy>=1.16.0,<1.19.1",
            "pandas>=1.0.4,<1.0.5",
            "scikit-learn>=0.21.3,<0.23.1"
            ]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except IOError:
    long_description = DESCRIPTION

setup(
      name='byted-'+NAME,
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
