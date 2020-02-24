#
# Common definitions for build.sh and BUILD. Keep this file compatible with
# Skylark, Python, bash, and GNU Make.
#

# Path to the Buildroot SDKs.
# XXX (markh): Pin the toolchain to a specific version once stable.
X86_64_SDK_PATH="/cb/tools/buildroot/dev/latest/sdk-dev-x86_64"
AARCH64_SDK_PATH="/cb/tools/buildroot/dev/latest/sdk-dev-aarch64"

# The gcc version string is used to specify the paths to gcc include directories
# (see cc_toolchain_config.bzl).
GCC_VERSION="8.3.0"
