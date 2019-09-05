CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O2
SRCFILE = src/gemm_kernel.S src/gemm_driver.c
INCFILE = src/gemm_kernel_irreg.c src/gemm_copy.c

default: ZGEMM.so CGEMM.so

ZGEMM.so: $(SRCFILE) $(INCFILE)
	$(CC) -DBlkDimK=128 -DBlkDimN=128 -DA_PR_BYTE=192 -DB_PR_ELEM=64 -DDOUBLE $(CCFLAGS) $(SRCFILE) -o $@
  
CGEMM.so: $(SRCFILE) $(INCFILE)
	$(CC) -DBlkDimK=256 -DBlkDimN=128 -DA_PR_BYTE=192 -DB_PR_ELEM=64 $(CCFLAGS) $(SRCFILE) -o $@

clean:
	rm -f *GEMM.so
