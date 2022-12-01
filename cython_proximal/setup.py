
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_proxim",
        ["cython_proxim.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cython_proxim',
    ext_modules=cythonize(ext_modules),
)
