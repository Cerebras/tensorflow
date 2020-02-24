#
# This file is used by BUILD to describe the Buildroot toolchains available to
# Bazel to compile TensorFlow. You should not need to edit this file
# frequently. Edit the toolchain.bzl file to change the toolchain version.
#
# This file is a slight modification of
# @local_config_cc//:cc_toolchain_config.bzl. There doesn't seem to a way to
# inherit or otherwise reuse it.
#

# Copyright 2019 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Starlark cc_toolchain configuration rule"""
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
     "action_config",
     "artifact_name_pattern",
     "env_entry",
     "env_set",
     "feature",
     "feature_set",
     "flag_group",
     "flag_set",
     "make_variable",
     "tool",
     "tool_path",
     "variable_with_value",
     "with_feature_set",
     )
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")


all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
    ACTION_NAMES.lto_backend,
]

all_cpp_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
]

preprocessor_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.clif_match,
]

codegen_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def buildroot_tool_path(ctx, name):
    return tool_path(
        name = name,
	path = ctx.attr.sdk_path + "/bin/" + ctx.attr.platform + "-" + name,
    )

def _impl(ctx):
    tool_paths = [
        buildroot_tool_path(ctx, "gcc"),
        buildroot_tool_path(ctx, "ld"),
        buildroot_tool_path(ctx, "ar"),
        buildroot_tool_path(ctx, "cpp"),
        buildroot_tool_path(ctx, "nm"),
  	buildroot_tool_path(ctx, "objdump"),
        tool_path(
            name = "gcov",
            path = "/bin/false",
        ),
        tool_path(
            name = "strip",
            path = "/usr/bin/strip",
        ),
    ]

    cxx_builtin_include_directories = [
	ctx.attr.sdk_path + "/include",
	ctx.attr.sdk_path + "/lib/gcc/" + ctx.attr.platform + "/" + ctx.attr.gcc_version + "/include",
	ctx.attr.sdk_path + "/lib/gcc/" + ctx.attr.platform + "/" + ctx.attr.gcc_version + "/include-fixed",
	ctx.attr.sdk_path + "/" + ctx.attr.platform + "/include/c++/" + ctx.attr.gcc_version,
	ctx.attr.sdk_path + "/" + ctx.attr.platform + "/sysroot/usr/include",
    ]

    action_configs = []

    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer"
    ]

    dbg_compile_flags = [
        "-g"
    ]

    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections"
    ]

    cxx_flags = [
        "-std=c++0x"
    ]

    link_flags = [
        # Buildroot does not support gold because it is not supported on all platforms.
	# Maybe we will enable it for x86_64 later.
	# "-fuse-ld=gold",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-pass-exit-codes",
        "-lstdc++",
        "-lm"
    ]

    opt_link_flags = [
        "-Wl,--gc-sections"
    ]

    unfiltered_compile_flags = [
        "-fno-canonical-system-headers",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\""
    ]

    targets_windows_feature = feature(
        name = "targets_windows",
        implies = ["copy_dynamic_libraries_to_binary"],
        enabled = True,
    )

    copy_dynamic_libraries_to_binary_feature = feature(name= "copy_dynamic_libraries_to_binary")

    gcc_env_feature = feature(
        name = "gcc_env",
        enabled = True,
        env_sets = [
            env_set (
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                env_entries = [
                    env_entry(key = "PATH", value = "NOT_USED")
                ],
            ),
        ],
    )

    windows_features = [
        targets_windows_feature,
        copy_dynamic_libraries_to_binary_feature,
        gcc_env_feature,
    ]

    
    coverage_feature = feature (
        name = "coverage",
        provides = ["profile"],
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group (flags = ["--coverage"]),
                ],
            ),
            flag_set (
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group (flags = ["--coverage"]),
                ],
            ),
        ],
    )


    supports_pic_feature = feature(
        name = "supports_pic",
        enabled = True,
    )
    supports_start_end_lib_feature = feature(
        name = "supports_start_end_lib",
        enabled = True,
    )

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = compile_flags)] if compile_flags else []),
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = dbg_compile_flags)] if dbg_compile_flags else []),
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = opt_compile_flags)] if opt_compile_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = cxx_flags)] if cxx_flags else []),
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = ([flag_group(flags = link_flags)] if link_flags else []),
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = ([flag_group(flags = opt_link_flags)] if opt_link_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    dbg_feature = feature(name = "dbg")

    opt_feature = feature(name = "opt")

    sysroot_feature = feature(
        name = "sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                  ACTION_NAMES.preprocess_assemble,
                  ACTION_NAMES.linkstamp_compile,
                  ACTION_NAMES.c_compile,
                  ACTION_NAMES.cpp_compile,
                  ACTION_NAMES.cpp_header_parsing,
                  ACTION_NAMES.cpp_module_compile,
                  ACTION_NAMES.cpp_module_codegen,
                  ACTION_NAMES.lto_backend,
                  ACTION_NAMES.clif_match,
                  ACTION_NAMES.cpp_link_executable,
                  ACTION_NAMES.cpp_link_dynamic_library,
                  ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                  flag_group(
                      flags = ["--sysroot=%{sysroot}"],
                      expand_if_available = "sysroot",
                  ),
                ],
            ),
        ],
    )

    fdo_optimize_feature = feature(
        name = "fdo_optimize",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                  flag_group(
                      flags = [
                        "-fprofile-use=%{fdo_profile_path}",
                        "-fprofile-correction",
                      ],
                      expand_if_available = "fdo_profile_path",
                  ),
                ],
            ),
        ],
        provides = ["profile"],
    )

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                  ACTION_NAMES.assemble,
                  ACTION_NAMES.preprocess_assemble,
                  ACTION_NAMES.linkstamp_compile,
                  ACTION_NAMES.c_compile,
                  ACTION_NAMES.cpp_compile,
                  ACTION_NAMES.cpp_header_parsing,
                  ACTION_NAMES.cpp_module_compile,
                  ACTION_NAMES.cpp_module_codegen,
                  ACTION_NAMES.lto_backend,
                  ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                  flag_group(
                      flags = ["%{user_compile_flags}"],
                      iterate_over = "user_compile_flags",
                      expand_if_available = "user_compile_flags",
                  ),
                ],
            ),
          ],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                  ACTION_NAMES.assemble,
                  ACTION_NAMES.preprocess_assemble,
                  ACTION_NAMES.linkstamp_compile,
                  ACTION_NAMES.c_compile,
                  ACTION_NAMES.cpp_compile,
                  ACTION_NAMES.cpp_header_parsing,
                  ACTION_NAMES.cpp_module_compile,
                  ACTION_NAMES.cpp_module_codegen,
                  ACTION_NAMES.lto_backend,
                  ACTION_NAMES.clif_match,
                ],
                flag_groups = ([flag_group(flags = unfiltered_compile_flags)] if unfiltered_compile_flags else []),
            ),
        ],
    )

    features = [
        supports_pic_feature,
        supports_start_end_lib_feature,
        coverage_feature,
        default_compile_flags_feature,
        default_link_flags_feature,
        fdo_optimize_feature,
        supports_dynamic_linker_feature,
        dbg_feature,
        opt_feature,
        user_compile_flags_feature,
        sysroot_feature,
        unfiltered_compile_flags_feature,
    ]

    artifact_name_patterns = [

    ]

    make_variables = []

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
        toolchain_identifier = ctx.attr.platform,
        host_system_name = "local",
        target_system_name = ctx.attr.platform,
        target_cpu = ctx.attr.cpu,
        target_libc = "glibc",
        compiler = "gcc-" + ctx.attr.gcc_version,
        abi_version = "gcc",
        abi_libc_version = "local",
        tool_paths = tool_paths,
        make_variables = make_variables,
        builtin_sysroot = "",
        cc_target_os = None,
    )

buildroot_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu" : attr.string(mandatory = True),
	"platform": attr.string(mandatory = True),
	"sdk_path": attr.string(mandatory = True),
	"gcc_version": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)
