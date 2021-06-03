from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='Autodidax extensions',
    ext_modules=cythonize("autodidax_ext.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)

