"""BUILD file for netlib BLAS.

This provides Reference BLAS 3.12.0 from netlib.org.

Available library targets:
  :single    - Single precision real (REAL)
  :double    - Double precision real (REAL*8)
  :complex   - Single precision complex (COMPLEX)
  :complex16 - Double precision complex (COMPLEX*16)
  :blas      - All precisions combined

Usage:
  deps = ["@blas//:double"]  # Only builds double precision
  deps = ["@blas//:blas"]    # All precisions

Note: Unlike CMake's BUILD_SINGLE/BUILD_DOUBLE/BUILD_COMPLEX/BUILD_COMPLEX16
options, Bazel only builds what you actually depend on. Just reference the
precision target you need - unused precisions won't be compiled.
"""

load("@rules_fortran//:defs.bzl", "fortran_binary", "fortran_library", "fortran_test")

# Auxiliary BLAS routines (required by all precisions)
ALLBLAS_SRCS = [
    "SRC/lsame.f",
    "SRC/xerbla.f",
    "SRC/xerbla_array.f",
]

# Single precision BLAS Level 1 routines
SBLAS1_SRCS = [
    "SRC/isamax.f",
    "SRC/sasum.f",
    "SRC/saxpy.f",
    "SRC/scopy.f",
    "SRC/sdot.f",
    "SRC/snrm2.f90",
    "SRC/srot.f",
    "SRC/srotg.f90",
    "SRC/sscal.f",
    "SRC/sswap.f",
    "SRC/sdsdot.f",
    "SRC/srotmg.f",
    "SRC/srotm.f",
]

# Single precision BLAS Level 2 routines
SBLAS2_SRCS = [
    "SRC/sgemv.f",
    "SRC/sgbmv.f",
    "SRC/ssymv.f",
    "SRC/ssbmv.f",
    "SRC/sspmv.f",
    "SRC/strmv.f",
    "SRC/stbmv.f",
    "SRC/stpmv.f",
    "SRC/strsv.f",
    "SRC/stbsv.f",
    "SRC/stpsv.f",
    "SRC/sger.f",
    "SRC/ssyr.f",
    "SRC/sspr.f",
    "SRC/ssyr2.f",
    "SRC/sspr2.f",
]

# Single precision BLAS Level 3 routines
SBLAS3_SRCS = [
    "SRC/sgemm.f",
    "SRC/ssymm.f",
    "SRC/ssyrk.f",
    "SRC/ssyr2k.f",
    "SRC/strmm.f",
    "SRC/strsm.f",
    "SRC/sgemmtr.f",
]

# Double precision BLAS Level 1 routines
DBLAS1_SRCS = [
    "SRC/idamax.f",
    "SRC/dasum.f",
    "SRC/daxpy.f",
    "SRC/dcopy.f",
    "SRC/ddot.f",
    "SRC/dnrm2.f90",
    "SRC/drot.f",
    "SRC/drotg.f90",
    "SRC/dscal.f",
    "SRC/dsdot.f",
    "SRC/dswap.f",
    "SRC/drotmg.f",
    "SRC/drotm.f",
]

# Double precision BLAS Level 2 routines
DBLAS2_SRCS = [
    "SRC/dgemv.f",
    "SRC/dgbmv.f",
    "SRC/dsymv.f",
    "SRC/dsbmv.f",
    "SRC/dspmv.f",
    "SRC/dtrmv.f",
    "SRC/dtbmv.f",
    "SRC/dtpmv.f",
    "SRC/dtrsv.f",
    "SRC/dtbsv.f",
    "SRC/dtpsv.f",
    "SRC/dger.f",
    "SRC/dsyr.f",
    "SRC/dspr.f",
    "SRC/dsyr2.f",
    "SRC/dspr2.f",
]

# Double precision BLAS Level 3 routines
DBLAS3_SRCS = [
    "SRC/dgemm.f",
    "SRC/dsymm.f",
    "SRC/dsyrk.f",
    "SRC/dsyr2k.f",
    "SRC/dtrmm.f",
    "SRC/dtrsm.f",
    "SRC/dgemmtr.f",
]

# Complex precision BLAS Level 1 routines
CBLAS1_SRCS = [
    "SRC/scabs1.f",
    "SRC/scasum.f",
    "SRC/scnrm2.f90",
    "SRC/icamax.f",
    "SRC/caxpy.f",
    "SRC/ccopy.f",
    "SRC/cdotc.f",
    "SRC/cdotu.f",
    "SRC/csscal.f",
    "SRC/crotg.f90",
    "SRC/cscal.f",
    "SRC/cswap.f",
    "SRC/csrot.f",
]

# Complex auxiliary routines (real BLAS called by complex)
CB1AUX_SRCS = [
    "SRC/isamax.f",
    "SRC/sasum.f",
    "SRC/saxpy.f",
    "SRC/scopy.f",
    "SRC/snrm2.f90",
    "SRC/sscal.f",
]

# Complex precision BLAS Level 2 routines
CBLAS2_SRCS = [
    "SRC/cgemv.f",
    "SRC/cgbmv.f",
    "SRC/chemv.f",
    "SRC/chbmv.f",
    "SRC/chpmv.f",
    "SRC/ctrmv.f",
    "SRC/ctbmv.f",
    "SRC/ctpmv.f",
    "SRC/ctrsv.f",
    "SRC/ctbsv.f",
    "SRC/ctpsv.f",
    "SRC/cgerc.f",
    "SRC/cgeru.f",
    "SRC/cher.f",
    "SRC/chpr.f",
    "SRC/cher2.f",
    "SRC/chpr2.f",
]

# Complex precision BLAS Level 3 routines
CBLAS3_SRCS = [
    "SRC/cgemm.f",
    "SRC/csymm.f",
    "SRC/csyrk.f",
    "SRC/csyr2k.f",
    "SRC/ctrmm.f",
    "SRC/ctrsm.f",
    "SRC/chemm.f",
    "SRC/cherk.f",
    "SRC/cher2k.f",
    "SRC/cgemmtr.f",
]

# Double complex precision BLAS Level 1 routines
ZBLAS1_SRCS = [
    "SRC/dcabs1.f",
    "SRC/dzasum.f",
    "SRC/dznrm2.f90",
    "SRC/izamax.f",
    "SRC/zaxpy.f",
    "SRC/zcopy.f",
    "SRC/zdotc.f",
    "SRC/zdotu.f",
    "SRC/zdscal.f",
    "SRC/zrotg.f90",
    "SRC/zscal.f",
    "SRC/zswap.f",
    "SRC/zdrot.f",
]

# Double complex auxiliary routines (real BLAS called by complex)
ZB1AUX_SRCS = [
    "SRC/idamax.f",
    "SRC/dasum.f",
    "SRC/daxpy.f",
    "SRC/dcopy.f",
    "SRC/dnrm2.f90",
    "SRC/dscal.f",
]

# Double complex precision BLAS Level 2 routines
ZBLAS2_SRCS = [
    "SRC/zgemv.f",
    "SRC/zgbmv.f",
    "SRC/zhemv.f",
    "SRC/zhbmv.f",
    "SRC/zhpmv.f",
    "SRC/ztrmv.f",
    "SRC/ztbmv.f",
    "SRC/ztpmv.f",
    "SRC/ztrsv.f",
    "SRC/ztbsv.f",
    "SRC/ztpsv.f",
    "SRC/zgerc.f",
    "SRC/zgeru.f",
    "SRC/zher.f",
    "SRC/zhpr.f",
    "SRC/zher2.f",
    "SRC/zhpr2.f",
]

# Double complex precision BLAS Level 3 routines
ZBLAS3_SRCS = [
    "SRC/zgemm.f",
    "SRC/zsymm.f",
    "SRC/zsyrk.f",
    "SRC/zsyr2k.f",
    "SRC/ztrmm.f",
    "SRC/ztrsm.f",
    "SRC/zhemm.f",
    "SRC/zherk.f",
    "SRC/zher2k.f",
    "SRC/zgemmtr.f",
]

# Compiler flags matching netlib BLAS defaults
# - See: https://flang.llvm.org/docs/OptionComparison.html
BLAS_COPTS = ["-O2"]

# Core libraries without error handlers (BLAS routines only, no xerbla)
# These are public so test programs can use them with their own xerbla implementations
fortran_library(
    name = "single_core",
    srcs = ["SRC/lsame.f"] + SBLAS1_SRCS + SBLAS2_SRCS + SBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
)

fortran_library(
    name = "double_core",
    srcs = ["SRC/lsame.f"] + DBLAS1_SRCS + DBLAS2_SRCS + DBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
)

fortran_library(
    name = "complex_core",
    srcs = ["SRC/lsame.f"] + CBLAS1_SRCS + CB1AUX_SRCS + CBLAS2_SRCS + CBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
)

fortran_library(
    name = "complex16_core",
    srcs = ["SRC/lsame.f"] + ZBLAS1_SRCS + ZB1AUX_SRCS + ZBLAS2_SRCS + ZBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
)

# Auxiliary library with error handlers (shared by all precisions)
fortran_library(
    name = "aux",
    srcs = ALLBLAS_SRCS,
    copts = BLAS_COPTS,
)

# Complete BLAS libraries with error handlers
# For normal use - includes default xerbla implementation
fortran_library(
    name = "single",
    srcs = SBLAS1_SRCS + SBLAS2_SRCS + SBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
    deps = [":aux"],
)

fortran_library(
    name = "double",
    srcs = DBLAS1_SRCS + DBLAS2_SRCS + DBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
    deps = [":aux"],
)

fortran_library(
    name = "complex",
    srcs = CBLAS1_SRCS + CBLAS2_SRCS + CBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":aux",
        ":single",
    ],
)

fortran_library(
    name = "complex16",
    srcs = ZBLAS1_SRCS + ZBLAS2_SRCS + ZBLAS3_SRCS,
    copts = BLAS_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":aux",
        ":double",
    ],
)

# Complete BLAS library with all precisions
fortran_library(
    name = "blas",
    visibility = ["//visibility:public"],
    deps = [
        ":single",
        ":double",
        ":complex",
        ":complex16",
    ],
)

# Test binaries (Level 1)
fortran_test(
    name = "sblat1",
    size = "small",
    srcs = ["TESTING/sblat1.f"],
    visibility = ["//visibility:public"],
    deps = [":single"],
)

fortran_test(
    name = "dblat1",
    size = "small",
    srcs = ["TESTING/dblat1.f"],
    visibility = ["//visibility:public"],
    deps = [":double"],
)

fortran_test(
    name = "cblat1",
    size = "small",
    srcs = ["TESTING/cblat1.f"],
    visibility = ["//visibility:public"],
    deps = [":complex"],
)

fortran_test(
    name = "zblat1",
    size = "small",
    srcs = ["TESTING/zblat1.f"],
    visibility = ["//visibility:public"],
    deps = [":complex16"],
)

# Test binaries (Level 2 - need input files)
# Test programs provide their own xerbla, so use *_core libraries
fortran_binary(
    name = "sblat2",
    srcs = ["TESTING/sblat2.f"],
    visibility = ["//visibility:public"],
    deps = [":single_core"],
)

fortran_binary(
    name = "dblat2",
    srcs = ["TESTING/dblat2.f"],
    visibility = ["//visibility:public"],
    deps = [":double_core"],
)

fortran_binary(
    name = "cblat2",
    srcs = ["TESTING/cblat2.f"],
    visibility = ["//visibility:public"],
    deps = [":complex_core"],
)

fortran_binary(
    name = "zblat2",
    srcs = ["TESTING/zblat2.f"],
    visibility = ["//visibility:public"],
    deps = [":complex16_core"],
)

# Test binaries (Level 3 - need input files)
# Test programs provide their own xerbla, so use *_core libraries
fortran_binary(
    name = "sblat3",
    srcs = ["TESTING/sblat3.f"],
    visibility = ["//visibility:public"],
    deps = [":single_core"],
)

fortran_binary(
    name = "dblat3",
    srcs = ["TESTING/dblat3.f"],
    visibility = ["//visibility:public"],
    deps = [":double_core"],
)

fortran_binary(
    name = "cblat3",
    srcs = ["TESTING/cblat3.f"],
    visibility = ["//visibility:public"],
    deps = [":complex_core"],
)

fortran_binary(
    name = "zblat3",
    srcs = ["TESTING/zblat3.f"],
    visibility = ["//visibility:public"],
    deps = [":complex16_core"],
)

# export test input files for use by examples
exports_files(
    glob(["TESTING/*.in"]),
    visibility = ["//visibility:public"],
)
