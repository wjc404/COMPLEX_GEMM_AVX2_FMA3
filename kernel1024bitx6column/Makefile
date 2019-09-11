CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O2
SRCFILE = src/gemm_kernel.S src/gemm_driver.c
INCFILE = src/gemm_kernel_irreg.c src/gemm_copy.c

#BlkDimK and BlkDimN must be exactly divisible by 12.

default: ZGEMM.so CGEMM.so

ZGEMM.so: $(SRCFILE) $(INCFILE)
	$(CC) -DBlkDimK=192 -DBlkDimN=192 -DA_PR_BYTE=256 -DB_PR_ELEM=48 -DDOUBLE $(CCFLAGS) $(SRCFILE) -o $@
  
CGEMM.so: $(SRCFILE) $(INCFILE)
	$(CC) -DBlkDimK=192 -DBlkDimN=192 -DA_PR_BYTE=256 -DB_PR_ELEM=48 $(CCFLAGS) $(SRCFILE) -o $@

clean:
	rm -f *GEMM.so
