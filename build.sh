#! /bin/bash
if [ ! -d "pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git
fi

echo "Installing gylm ..."
mkdir -p build
cd build
cmake .. -DPYBIND11_PYTHON_VERSION=3.6 && make -j 4 && make install
cd ..

