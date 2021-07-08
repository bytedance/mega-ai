# coding: utf-8

from setuptools import find_packages, setup

import io
import os

# Package meta information
NAME = 'magellan-ai'
DESCRIPTION = 'Some computing tools with machine learning'
AUTHOR = 'huang ning'

# The dependent version required for the project to run
REQUIRES = ["numpy>=1.16.0,<1.19.1",
            "pandas>=1.0.4,<1.2.0",
            "scikit-learn>=0.21.3,<0.23.1",
            "six>=1.10.0,<2.0.0",
            "tensorflow>=2.2.0,<2.4.0",
            "tensorflow-serving-api>=2.2.0",
            "pyarrow>=1.0.0, <=4.0.0",
            "openpyxl>=3.0.0, <4.0.0",
            "xlrd>=1.0.0, <2.0.0",
            "keras>=2.4.3, <3.0.0",
            "gensim>=1.0.0",
            "wheel>=0.23.0",
            "Cython>=0.20.2",
            "six>=1.7.3",
            "gensim>=1.0.0",
            "scipy>=0.15.0",
            "psutil>=2.1.1",
            "networkx>=2.0",
            ""
            ]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except IOError:
    long_description = DESCRIPTION

setup(
      name='byted-'+NAME,
      version='1.4.4',
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
