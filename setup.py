from setuptools import setup, find_packages
setup(name='magellan-ml',
      version='0.0.2',
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
            'numpy'
      ],
      zip_safe=False)