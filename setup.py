#!/usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize

setup(name='TED app', ext_modules=cythonize("*.pyx"), zip_safe=False)
