# coding: utf-8

from setuptools import find_packages, setup

import io
import os

# 包元信息
NAME = 'magellan-ai'
DESCRIPTION = 'Some computing tools with machine learning, python 3.6+.'
AUTHOR = 'huangning'

# 项目运行需要的依赖
REQUIRES = ["numpy>=1.16.0,<1.19.1",
            "pandas>=1.0.4,<1.0.5",
            "scikit-learn>=0.21.3,<0.23.1",
            "six>=1.11.0,<2.0.0"
            "tensorflow>=2.0.0,<2.4.0"
            ]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except IOError:
    long_description = DESCRIPTION

setup(
      name='byted-'+NAME,
      version='1.2.1',
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author=AUTHOR,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      packages=find_packages(),
      install_requires=REQUIRES,
      python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
      include_package_data=True
)
