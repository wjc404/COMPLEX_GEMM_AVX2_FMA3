# COMPLEX_GEMM_AVX2_FMA3
Heavily-optimized cgemm and zgemm subroutines for large matrices(dim 3000-30000), using avx2 and fma3 instructions, with performance comparable to MKL2018, able to achieve >95% theoretical performance in serial executions. 

interface: fortran, 32-bit integer

Tuned parameters (see Makefile):

    Core i9 9900K: BlkDimN = 192, B_PR_ELEM = 40, A_PR_BYTE = 192 or 256; BlkDimK: 128 for ZGEMM, 256 for CGEMM.
    Ryzen 7 3700X: BlkDimN = 96,  B_PR_ELEM = 24, A_PR_BYTE = 256; BlkDimK: 128 for ZGEMM, 256 for CGEMM.
