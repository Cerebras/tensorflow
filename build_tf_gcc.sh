#!/bin/bash

source ~/.bashrc

set -e
set -x

#cd $(dirname $(realpath $0))/pytorch/xla/third_party/tensorflow


unset CC
unset CXX

export CC=/opt/gcc-6.3.0/bin/gcc
export CXX=/opt/gcc-6.3.0/bin/g++

if [ "$1" == "clean" ]; then
  bazel --output_user_root=/spare/bazel-cache-gcc clean
else
  bazel --output_user_root=/spare/bazel-cache-gcc build --config=mkl --strip=never --copt=-fdiagnostics-color //tensorflow/tools/pip_package:build_pip_package
fi

