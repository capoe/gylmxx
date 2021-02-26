import sys
import platform
from setuptools import setup, find_packages, Extension
from subprocess import getoutput
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def make_cxx_extensions():
    srcs = [
        "./gylm/cxx/gylm.cpp",
        "./gylm/cxx/kernel.cpp",
        "./gylm/cxx/ylm.cpp",
        "./gylm/cxx/bindings.cpp",
        "./gylm/cxx/celllist.cpp",
        "./gylm/cxx/soapgto.cpp"]
    incdirs = [
        "./gylm/cxx",
        get_pybind_include(),
        get_pybind_include(user=True)]
    cpp_extra_link_args = []
    cpp_extra_compile_args = ["-std=c++11", "-O3"]
    c_extra_compile_args = ["-std=c99", "-O3"]
    return [
        Extension(
            'gylm._gylm',
            srcs,
            include_dirs=incdirs,
            language='c++',
            extra_compile_args=cpp_extra_compile_args + ["-fvisibility=hidden"],  # the -fvisibility flag is needed by pybind11
            extra_link_args=cpp_extra_link_args,
        )
    ]

def get_description():
    return "This library implements Gnl-Ylm-type convolutions as a superset of SOAP, with additional control over spatial frequency damping and radial discounting"

if __name__ == "__main__":
    setup(
        name="gylm",
        version="0.0.0",
        author="capoe",
        author_email="carl.poelking@floatlab.io",
        url="https://github.com/capoe/gylmxx",
        description="Implementation of Gnl-Ylm-type 3d convolutional descriptors",
        long_description=get_description(),
        packages=find_packages(),
        setup_requires=['pybind11>=2.4'],
        install_requires=['pybind11>=2.4', "numpy", "scipy", "scikit-learn"],
        include_package_data=True,
        ext_modules=make_cxx_extensions(),
        license="Apache License 2.0",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
        keywords="3d convolutional descriptors chemical machine learning",
        python_requires=">=3.7",
    )

