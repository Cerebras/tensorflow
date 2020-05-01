#
# Common definitions for build.sh and BUILD. Keep this file compatible with
# Skylark, Python, bash, and GNU Make.
#

# Path to the Buildroot SDKs.
# XXX (markh): Pin the toolchain to a specific version once stable.
X86_64_SDK_PATH="/cb/toolchains/buildroot/monolith-default/latest/sdk-default-x86_64"
# XXX (markh): Maybe eventually.
AARCH64_SDK_PATH="/cb/toolchains/buildroot/monolith-default/latest/sdk-default-aarch64"

# The gcc version string is used to specify the paths to gcc include directories
# (see cc_toolchain_config.bzl).
GCC_VERSION="8.3.0"
