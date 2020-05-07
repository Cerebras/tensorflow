#!/bin/bash
#
# This example build script builds tensorflow using the toolchain specified in
# third_party/toolchains/cerebras/toolchain.bzl.
#
# TODO (markh): Consider moving the build to Buildroot so that the toolchain is
# self-contained. Right now, we use the last deployed version to build
# TensorFlow, then install it in the next deployed version.
#
# owner: markh (Mark Huang)

set -x

ARCH=${ARCH:-x86_64}

. ./third_party/toolchains/cerebras/toolchain.bzl

if [ "$ARCH" = "x86_64" ] ; then
    SDK_PATH=$X86_64_SDK_PATH
    CC_OPT_FLAGS="-march=broadwell -Wno-sign-compare"
    BUILD_FLAGS="--cpu=k8 --config=mkl"
else
    SDK_PATH=$AARCH64_SDK_PATH
    CC_OPT_FLAGS="-Wno-sign-compare"
    BUILD_FLAGS="--cpu=arm64-v8a"
fi

# SW-21674: The path cannot be a symlink or else bazel will complain because
# this path is added to cxx_builtin_include_directories as-is, but bazel
# resolves the full paths to detected dependencies and does string comparisons,
# not same-file comparisons.
if [ "$SDK_PATH" != $(realpath "$SDK_PATH") ] ; then
    echo "$SDK_PATH -> $(realpath "$SDK_PATH") cannot be a symlink! See SW-21674."
    exit 1
fi

# Configure the build.
PATH=$SDK_PATH/usr/bin:$PATH \
PYTHON_BIN_PATH=$SDK_PATH/usr/bin/python3 \
PYTHON_LIB_PATH=$($SDK_PATH/usr/bin/python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
TF_ENABLE_XLA=1 \
TF_NEED_OPENCL_SYCL=0 \
TF_NEED_ROCM=0 \
TF_NEED_CUDA=0 \
TF_DOWNLOAD_CLANG=0 \
TF_NEED_MPI=0 \
CC_OPT_FLAGS="$CC_OPT_FLAGS" \
TF_SET_ANDROID_WORKSPACE=0 \
./configure

$SDK_PATH/usr/bin/bazel \
    --output_user_root=/scratch/tensorflow \
    build \
    --verbose_failures \
    --config=cerebras \
    --config=opt \
    $BUILD_FLAGS \
    //tensorflow/tools/pip_package:build_pip_package \
    && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package .
rc=$?

$SDK_PATH/usr/bin/bazel shutdown

exit $rc
