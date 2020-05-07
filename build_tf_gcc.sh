#!/bin/bash

source ~/.bashrc

set -e
set -x

#cd $(dirname $(realpath $0))/pytorch/xla/third_party/tensorflow


#unset CC
#unset CXX

export CC=/cb/toolchains/buildroot/monolith-default/chriso/sdk-default-x86_64/usr/bin/x86_64-buildroot-linux-gnu-gcc
export CXX=/cb/toolchains/buildroot/monolith-default/chriso/sdk-default-x86_64/usr/bin/x86_64-buildroot-linux-gnu-g++

HOST="$(hostname)"

if [ "$1" == "clean" ]; then
  bazel --output_user_root=/spare/bazel-cache-${HOST}-toolchain clean
else
  bazel --output_user_root=/spare/bazel-cache-${HOST}-toolchain build --config=mkl --strip=never --copt=-fdiagnostics-color //tensorflow/tools/pip_package:build_pip_package
fi

