#
# This file describes the Buildroot toolchains available to Bazel to compile
# TensorFlow. You should not need to edit this file frequently. Edit the
# toolchain.bzl file to change the toolchain version.
#
# The appropriate toolchain is selected by specifying --crosstool_top and --cpu
# on the bazel command line:
#
# bazel build \
#     --crosstool_top=//third_party/toolchains/cerebras:buildroot \
#     --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#     --cpu=k8 \
#     ...
#
# Or in .bazelrc (in which case you only need to specify --config=cerebras and
# --cpu on the bazel command line):
#
# build:cerebras --crosstool_top=//third_party/toolchains/cerebras:buildroot
# build:cerebras --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
#

package(default_visibility = ["//visibility:public"])

load(":buildroot_toolchain_config.bzl", "buildroot_toolchain_config")
load(":toolchain.bzl", "X86_64_SDK_PATH", "AARCH64_SDK_PATH", "GCC_VERSION")

cc_toolchain_suite(
    # The name of the toolchain suite. Keep this in sync with .bazelrc, e.g.
    # --crosstool_top=//third_party/toolchains/cerebras:buildroot
    #                                                   ^^^^^^^^^
    name = "buildroot",
    # cpu:cc_toolchain.name mapping. The cpu names are Bazel conventions. Keep
    # each toolchain name in sync with the cc_toolchain() definitions below.    
    toolchains = {
        "k8": ":x86_64-buildroot-linux-gnu",
	"arm64-v8a": "aarch64-buildroot-linux-gnu",
    },
)

filegroup(name = "empty")

#
# x86_64
#

cc_toolchain(
    # Keep this in sync with cc_toolchain_suite.toolchains above.
    name = "x86_64-buildroot-linux-gnu",
    # Keep this in sync with buildroot_toolchain_config.platform below.
    toolchain_identifier = "x86_64-buildroot-linux-gnu",
    # Keep this in sync with buildroot_toolchain_config.name below.
    toolchain_config = ":x86_64-buildroot-linux-gnu-config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    ar_files = ":empty",
    strip_files = ":empty",
    supports_param_files = False,
)

buildroot_toolchain_config(
    # Keep this in sync with cc_toolchain.toolchain_config above.
    name = "x86_64-buildroot-linux-gnu-config",
    # Keep this in sync with cc_toolchain_suite.toolchains above.
    cpu = "k8",
    # Keep this in sync with cc_toolchain.toolchain_identifier above.
    platform = "x86_64-buildroot-linux-gnu",
    sdk_path = X86_64_SDK_PATH,
    gcc_version = GCC_VERSION,
)

#
# aarch64
#

cc_toolchain(
    # Keep this in sync with cc_toolchain_suite.toolchains above.
    name = "aarch64-buildroot-linux-gnu",
    # Keep this in sync with buildroot_toolchain_config.platform below.
    toolchain_identifier = "aarch64-buildroot-linux-gnu",
    # Keep this in sync with buildroot_toolchain_config.name below.
    toolchain_config = ":aarch64-buildroot-linux-gnu-config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    ar_files = ":empty",
    strip_files = ":empty",
    supports_param_files = False,
)

buildroot_toolchain_config(
    # Keep this in sync with cc_toolchain.toolchain_config above.
    name = "aarch64-buildroot-linux-gnu-config",
    # Keep this in sync with cc_toolchain_suite.toolchains above.
    cpu = "arm64-v8a",
    # Keep this in sync with cc_toolchain.toolchain_identifier above.
    platform = "aarch64-buildroot-linux-gnu",
    sdk_path = AARCH64_SDK_PATH,
    gcc_version = GCC_VERSION,
)
