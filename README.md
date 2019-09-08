# COMPLEX_GEMM_AVX2_FMA3
cgemm and zgemm subroutines for large matrices, using avx2 and fma3 instructions, with performance comparable to MKL2018

interface: fortran, 32-bit integer

Tuned parameters (see Makefile):

    Core i9-9900K: BlkDimN = 192, B_PR_ELEM = 40, A_PR_BYTE = 192 or 256; BlkDimK: 128 for ZGEMM, 256 for CGEMM.
