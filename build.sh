#! /bin/bash
if [ ! -d "pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git
fi
py_version=$(python -c "import sys; print('.'.join(sys.version.split()[0].split('.')[0:2]))")

echo "Installing gylm using python=${py_version}"
mkdir -p build
cd build
cmake .. \
    -DPYBIND11_PYTHON_VERSION=${py_version} \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    && make -j 4 && make install
cd ..

