from setuptools import Extension, setup

import pybind11
from pybind11.setup_helpers import build_ext


ext_modules = [
    Extension(
        "qmodel.data.cpp.permute",
        ["qmodel/data/cpp/permute.cpp"],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-fopenmp", "-std=c++11"],
        extra_link_args=["-fopenmp"],
    ),
]


setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
