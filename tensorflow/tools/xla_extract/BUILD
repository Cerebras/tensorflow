package(default_visibility = ["//tensorflow:__subpackages__"])

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//tensorflow/tools/xla_extract:internal"],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "tf_graph_to_xla_lib",
    srcs = ["tf_graph_to_xla_lib.cc"],
    hdrs = ["tf_graph_to_xla_lib.h"],
    deps = [
       "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/compiler/jit:xla_cpu_device",
        # "//tensorflow/compiler/jit:xla_compiler",
        "//tensorflow/compiler/xla/service/interpreter:compiler",
	    "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/service:despecializer",
        # "//tensorflow/compiler/xla/client:client_library", # friends only
        # "//tensorflow/compiler/xla/service", # friends only
        # "//tensorflow/compiler/xla/service:cpu_plugin",
        # "@local_config_tf//:libtensorflow_framework",
    ],
)

tf_cc_binary(
    name = "tf_graph_to_xla",
    srcs = ["tf_graph_to_xla.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_graph_to_xla_lib",
    ],
)

py_library(
    name = "xla_extract",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)