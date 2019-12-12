package(default_visibility = ['//visibility:public'])

filegroup(
  name = 'gcc',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-gcc',
  ],
)

filegroup(
  name = 'ar',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-ar',
  ],
)

filegroup(
  name = 'ld',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-ld',
  ],
)

filegroup(
  name = 'nm',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-nm',
  ],
)

filegroup(
  name = 'objcopy',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-objcopy',
  ],
)

filegroup(
  name = 'objdump',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-objdump',
  ],
)

filegroup(
  name = 'strip',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-strip',
  ],
)

filegroup(
  name = 'as',
  srcs = [
    'bin/powerpc64le-buildroot-linux-gnu-as',
  ],
)

filegroup(
  name = 'compiler_pieces',
  srcs = glob([
    'powerpc64le-buildroot-linux-gnu/**',
    'libexec/**',
    'lib64/gcc/powerpc64le-buildroot-linux-gnu/**',
    'include/**',
  ]),
)

filegroup(
  name = 'compiler_components',
  srcs = [
    ':gcc',
    ':ar',
    ':ld',
    ':nm',
    ':objcopy',
    ':objdump',
    ':strip',
    ':as',
  ],
)
