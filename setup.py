import os
from distutils.core import setup, Extension
import numpy
import shutil
import glob


sources = ['longdouble_conv/longdouble_pywrap.c','longdouble_conv/longdouble.c']

ext=Extension("longdouble_conv._longdouble_pywrap",
              sources,
              extra_compile_args = ['-std=gnu99'])


setup(name="longdouble_conv", 
      version="0.1",
      description="Accurate longdouble conversions",
      license = "GPL",
      author="Eli Rykoff",
      author_email="erykoff@gmail.com",
      ext_modules=[ext],
      include_dirs=[numpy.get_include()],
      packages=['longdouble_conv'])


