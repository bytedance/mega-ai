from setuptools import setup, find_packages
setup(name='magellan-ai',
      version='0.0.1',
      description='Some computing tools with machine learning, python 3.6+.',
      url='http://github.com/zhangliang605/magellan_ml',
      author='zhangliang.thanks',
      author_email='zhangliang.thanks@bytedance.com',
      license='MIT',
      packages=find_packages(),
      include_package_data = True,
      platforms = 'any',
      python_requires='>=3',
      install_requires = [
            'pandas',
            'numpy',
            'xgboost'
      ],
      zip_safe=False)