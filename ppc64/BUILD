# This file is expanded from a template by cuda_configure.bzl
# Update cuda_configure.bzl#verify_build_defines when adding new variables.

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

toolchain(
    name = "toolchain-target-ppc64",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:ppc",
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "ppc64|compiler": ":cc-compiler-ppc64",
        "k8": ":cc-compiler-local",
        "powerpc64le": ":cc-compiler-ppc64",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":ppc64_linux_all_files",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":ppc64_linux_all_files",
    objcopy_files = ":empty",
    strip_files = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
    toolchain_config = ":cc-compiler-local-config",
    toolchain_identifier = "local_linux",
)

cc_toolchain_config(
    name = "cc-compiler-local-config",
    builtin_include_directories = [
        "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include",
        "/opt/rh/devtoolset-7/root/usr/include/c++/7",
        "/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include",
        "/opt/rh/devtoolset-7/root/usr/include/c++/7/x86_64-redhat-linux/",
        "/usr/local/include",
        "/usr/include",
        "/usr/local/cuda-10.0/targets/x86_64-linux/include",
        "/usr/local/cuda-10.0/include",
        "/usr/local/cuda-10.0/extras/CUPTI/include",
        "/usr/local/cuda-10.0/include",
    ],
    cpu = "local",
    extra_no_canonical_prefixes_flags = ["-fno-canonical-system-headers"],
    host_compiler_path = "/opt/rh/devtoolset-7/root/usr/bin/gcc",
    host_compiler_prefix = "/opt/rh/devtoolset-7/root/usr/bin",
    host_compiler_warnings = [],
    host_unfiltered_compile_flags = [],
    linker_bin_path = "/opt/rh/devtoolset-7/root/usr/bin",
)

cc_toolchain(
    name = "cc-compiler-ppc64",
    all_files = ":ppc64_linux_all_files",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":ppc64_linux_all_files",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":cc-compiler-local-ppc64",
    toolchain_identifier = "local_ppc64",
)

cc_toolchain_config(
    name = "cc-compiler-local-ppc64",
    builtin_include_directories = [
        "/srv/nfs/vasu-data/ws/buildroot-private/output/host/powerpc64le-buildroot-linux-gnu/sysroot/usr/include",
        "/srv/nfs/vasu-data/ws/buildroot-private/output/host/lib/gcc/powerpc64le-buildroot-linux-gnu/7.3.0/include",
        "/srv/nfs/vasu-data/ws/buildroot-private/output/host/powerpc64le-buildroot-linux-gnu/include/c++/7.3.0",
        "/srv/nfs/vasu-data/ws/buildroot-private/output/host/lib/gcc/powerpc64le-buildroot-linux-gnu/7.3.0/include-fixed",
        "/usr/local/include",
        "/usr/include",
        "/usr/local/cuda-10.0/targets/x86_64-linux/include",
        "/usr/local/cuda-10.0/include",
        "/usr/local/cuda-10.0/extras/CUPTI/include",
        "/usr/local/cuda-10.0/include",
    ],
    cpu = "powerpc64le",
    extra_no_canonical_prefixes_flags = ["-fno-canonical-system-headers"],
    host_compiler_path = "/srv/nfs/vasu-data/ws/buildroot-private/output/host/bin/powerpc64le-buildroot-linux-gnu-gcc",
    host_compiler_prefix = "/srv/nfs/vasu-data/ws/buildroot-private/output/host/bin",
    host_compiler_warnings = [],
    host_unfiltered_compile_flags = [],
    linker_bin_path = "/srv/nfs/vasu-data/ws/buildroot-private/output/host/bin",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "ppc64_linux_all_files",
    srcs = [
        "@toolchain_target_ppc64//:compiler_pieces",
    ],
)

