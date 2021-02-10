from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='neuralalgo',
      py_modules=['neuralalgo'],
      install_requires=[
          'torch',
          'numpy'
      ],
)
