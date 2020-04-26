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
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library_cc")

cc_library(
    name = "wse_hlo",
    srcs = ["wse_hlo.cc"],
    hdrs = ["wse_hlo.h", "utils.h"],
    copts = [ "-g", "-O0" ],
    deps = [
       "//tensorflow/core:framework",
       "//tensorflow/core:lib",
       "//tensorflow/core:ops",
       "//tensorflow/compiler/xla:xla_proto_cc",
       "//tensorflow/compiler/tf2xla:host_compute_metadata_proto_cc",
       "//tensorflow/compiler/xla/service:hlo_profile_printer_data_cc",
       "//tensorflow/compiler/xla/service:hlo_execution_profile_data_cc",
       "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
       "//tensorflow/compiler/xla/service:compiler",
       "//tensorflow/compiler/xla/client:compile_only_client",
       "//tensorflow/compiler/xla/client:client_library",
       "//tensorflow/compiler/xla/client:local_client",
       "//tensorflow/compiler/xla/service:algebraic_simplifier",
       "//tensorflow/compiler/xla/service:flatten_call_graph",
       "//tensorflow/compiler/xla/service:hlo_constant_folding",
       "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
       "//tensorflow/compiler/xla/service:layout_assignment",
       "//tensorflow/compiler/xla/service:reshape_mover",
       "//tensorflow/compiler/xla/service:while_loop_simplifier",
       "//tensorflow/compiler/xla/service:hlo_cse",
       "//tensorflow/compiler/xla/service:hlo_dce",
    ],
)

cc_library(
    name = "tf_graph_to_xla_lib",
    srcs = ["tf_graph_to_xla_lib.cc"],
    hdrs = ["tf_graph_to_xla_lib.h"],
    copts = [ "-g", "-O0" ],
    deps = [
       "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/compiler/jit:xla_cpu_device",
        "//tensorflow/compiler/xla/service/interpreter:compiler",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/service:despecializer",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        ":wse_hlo",
    ],
)

tf_cc_binary(
    name = "tf_graph_to_xla",
    srcs = ["tf_graph_to_xla.cc"],
    copts = [ "-g", "-O0" ],
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
