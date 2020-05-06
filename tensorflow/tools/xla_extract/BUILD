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

        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:svd",

        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",

        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:logdet",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:pooling",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/compiler/xla/client/lib:sorting",
        "//tensorflow/compiler/xla/client/lib:tridiagonal",
        "//tensorflow/compiler/xla/rpc:grpc_stub",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_server",
        "//tensorflow/compiler/xrt/cc:xrt_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:data_flow",
        "//tensorflow/stream_executor:stream_executor_impl",
        "//tensorflow/tools/xla_extract:wse_hlo",
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
