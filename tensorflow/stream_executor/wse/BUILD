# Description:
#   WSE-platform specific StreamExecutor support code.

load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")
# load("@local_config_wse//wse:build_defs.bzl", "if_wse_is_configured")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

# Filegroup used to collect source files for the dependency check.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "wse_platform_id",
    srcs = ["wse_platform_id.cc"],
    hdrs = ["wse_platform_id.h"],
    deps = ["//tensorflow/stream_executor:platform"],
)

cc_library(
    name = "all_runtime",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps =[
        ":wse_platform_id",
    ],
    alwayslink = 1,
)
