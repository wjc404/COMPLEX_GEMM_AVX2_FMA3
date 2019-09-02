# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h> //AVX2
# include <omp.h>

#ifdef DOUBLE
 #define FLOAT double
 #define BlkDimM 6
 #define CNAME zgemm_
#else
 #define FLOAT float
 #define BlkDimM 12
 #define CNAME cgemm_
#endif

# include "gemm_copy.c"
# include "gemm_kernel_irreg.c"
extern void gemmblkregccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc);//carry >90% gemm calculations
extern void gemmblktailccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int mdim);
extern void timedelay();//produce nothing besides a delay(~3 us), with no system calls
static void synproc(int tid,int threads,int *workprogress){//workprogress[] must be shared among all threads
  int waitothers,ctid,temp;
  workprogress[16*tid]++;
  temp=workprogress[16*tid];
  for(waitothers=1;waitothers;timedelay()){
    waitothers=0;
    for(ctid=0;ctid<threads;ctid++){
      if(workprogress[16*ctid]<temp) waitothers = 1;
    }
  }
}//this function is for synchronization of threads before/after load_abuffer
static void load_abuffer_ac(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_c(aheadpos+i*BlkDimM*2,abuffer+i*BlkDimM*2*BlkDimK,LDA);
  load_tail_a_c(aheadpos+i*BlkDimM*2,abuffer+i*BlkDimM*2*BlkDimK,LDA,EdgeM);
}
static void load_abuffer_ar(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_r(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*BlkDimK,LDA);
  load_tail_a_r(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*BlkDimK,LDA,EdgeM);
}
static void load_abuffer_ah(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_h(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*BlkDimK,LDA);
  load_tail_a_h(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*BlkDimK,LDA,EdgeM);
}
static void load_abuffer_irregk_ac(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_c(aheadpos+i*BlkDimM*2,abuffer+i*BlkDimM*2*kdim,LDA,kdim);
  load_irreg_a_c(aheadpos+i*BlkDimM*2,abuffer+i*BlkDimM*2*kdim,LDA,EdgeM,kdim);
}
static void load_abuffer_irregk_ar(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_r(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*kdim,LDA,kdim);
  load_irreg_a_r(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*kdim,LDA,EdgeM,kdim);
}
static void load_abuffer_irregk_ah(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_h(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*kdim,LDA,kdim);
  load_irreg_a_h(aheadpos+i*BlkDimM*2*LDA,abuffer+i*BlkDimM*2*kdim,LDA,EdgeM,kdim);
}
static void gemmcolumn(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkregccc(abuffer+MCT*2*BlkDimK,bblk,cheadpos+MCT*2,LDC);
    MCT+=BlkDimM;
  }
  gemmblktailccc(abuffer+MCT*2*BlkDimK,bblk,cheadpos+MCT*2,LDC,EdgeM);
}
static void gemmcolumnirregn(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC,int ndim){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkirregnccc(abuffer+MCT*2*BlkDimK,bblk,cheadpos+MCT*2,LDC,ndim);
    MCT+=BlkDimM;
  }
  gemmblkirregccc(abuffer+MCT*2*BlkDimK,bblk,cheadpos+MCT*2,LDC,EdgeM,ndim,BlkDimK);
}
static void gemmcolumnirregk(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC,int kdim){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkirregkccc(abuffer+MCT*2*kdim,bblk,cheadpos+MCT*2,LDC,kdim);
    MCT+=BlkDimM;
  }
  gemmblkirregccc(abuffer+MCT*2*kdim,bblk,cheadpos+MCT*2,LDC,EdgeM,BlkDimN,kdim);
}
static void gemmcolumnirreg(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,int ndim){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkirregccc(abuffer+MCT*2*kdim,bblk,cheadpos+MCT*2,LDC,BlkDimM,ndim,kdim);
    MCT+=BlkDimM;
  }
  gemmblkirregccc(abuffer+MCT*2*kdim,bblk,cheadpos+MCT*2,LDC,EdgeM,ndim,kdim);
}
void CNAME(char *transa,char *transb,int *m,int *n,int *k,FLOAT *alpha,FLOAT *a,int *lda,FLOAT *bstart,int *ldb,FLOAT *beta,FLOAT *cstart,int *ldc){
//assume column-major storage with arguments passed by addresses (FORTRAN style)
//a:matrix with m rows and k columns if transa=N
//b:matrix with k rows and n columns if transb=N
//c:product matrix with m rows and n columns
 const int M = *m;/* const int N = *n; */const int K = *k;
 const int LDA = *lda;const int LDB = *ldb;const int LDC=*ldc;
 const char TRANSA = *transa;const char TRANSB = *transb;
 const int BlksM = (M-1)/BlkDimM+1;const int EdgeM = M-(BlksM-1)*BlkDimM;//the m-dim of edges
 const int BlksK = (K-1)/BlkDimK+1;const int EdgeK = K-(BlksK-1)*BlkDimK;//the k-dim of edges
 int *workprogress, *cchunks;const int numthreads=omp_get_max_threads();int i; //for parallel execution
 //cchunk[] for dividing tasks, workprogress[] for recording the progresses of all threads and synchronization.
 //synchronization is necessary here since abuffer[] is shared between threads.
 //if abuffer[] is thread-private, the bandwidth of memory will limit the performance.
 //synchronization by openmp functions can be expensive, so handcoded funcion (synproc) is used instead.
 FLOAT *abuffer; //abuffer[]: store 256 columns of matrix a
 cmultbeta(cstart,LDC,M,(*n),beta);//limited by memory bendwidth so no need for parallel execution
 if(alpha[0]==0.0 && alpha[1]==0.0) return;
//then do C+=alpha*AB
  abuffer = (FLOAT *)aligned_alloc(4096,(BlkDimM*2*BlkDimK*BlksM)*sizeof(FLOAT));
  workprogress = (int *)calloc(20*numthreads,sizeof(int));
  cchunks = (int *)malloc((numthreads+1)*sizeof(int));
  for(i=0;i<=numthreads;i++) cchunks[i]=(*n)*i/numthreads;
#pragma omp parallel
 {
  int tid = omp_get_thread_num();
  FLOAT *c = cstart + LDC * 2 * cchunks[tid];
  FLOAT *b;
  if(TRANSB=='N' || TRANSB=='n') b = bstart + LDB * 2 * cchunks[tid];
  else b = bstart + 2 * cchunks[tid];
  const int N = cchunks[tid+1]-cchunks[tid];
  const int BlksN = (N-1)/BlkDimN+1; const int EdgeN = N-(BlksN-1)*BlkDimN;//the n-dim of edges
  int BlkCtM,BlkCtN,BlkCtK,MCT,NCT,KCT;//loop counters over blocks
  //MCT,NCT and KCT are used to locate the current position of matrix blocks
  FLOAT *bblk = (FLOAT *)aligned_alloc(4096,(BlkDimN*BlkDimK*2)*sizeof(FLOAT)); //thread-private bblk[]

    if(tid==0){
     if(TRANSA=='N')      load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
     else if(TRANSA=='C') load_abuffer_irregk_ah(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
     else                 load_abuffer_irregk_ar(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    }
    synproc(tid,numthreads,workprogress);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     if(TRANSB=='N')      load_irreg_b_c(b+NCT*LDB*2,bblk,LDB,BlkDimN,EdgeK,alpha);
     else if(TRANSB=='C') load_irreg_b_h(b+NCT*2,bblk,LDB,BlkDimN,EdgeK,alpha);
     else                 load_irreg_b_r(b+NCT*2,bblk,LDB,BlkDimN,EdgeK,alpha);
     gemmcolumnirregk(abuffer,bblk,c+NCT*LDC*2,BlksM,EdgeM,LDC,EdgeK);
    }
    NCT=BlkDimN*(BlksN-1);
    if(TRANSB=='N')      load_irreg_b_c(b+NCT*LDB*2,bblk,LDB,EdgeN,EdgeK,alpha);
    else if(TRANSB=='C') load_irreg_b_h(b+NCT*2,bblk,LDB,EdgeN,EdgeK,alpha);
    else                 load_irreg_b_r(b+NCT*2,bblk,LDB,EdgeN,EdgeK,alpha);
    gemmcolumnirreg(abuffer,bblk,c+NCT*LDC*2,BlksM,EdgeM,LDC,EdgeK,EdgeN);
    synproc(tid,numthreads,workprogress);//before updating abuffer, the master thread need to wait here until all child threads finish calculation with current abuffer
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0){
      if(TRANSA=='N')      load_abuffer_ac(a+KCT*LDA*2,abuffer,LDA,BlksM,EdgeM);
      else if(TRANSA=='C') load_abuffer_ah(a+KCT*2,abuffer,LDA,BlksM,EdgeM);
      else                 load_abuffer_ar(a+KCT*2,abuffer,LDA,BlksM,EdgeM);
     }
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      if(TRANSB=='N')      load_reg_b_c(b+(NCT*LDB+KCT)*2,bblk,LDB,alpha);
      else if(TRANSB=='C') load_reg_b_h(b+(KCT*LDB+NCT)*2,bblk,LDB,alpha);
      else                 load_reg_b_r(b+(KCT*LDB+NCT)*2,bblk,LDB,alpha);
      gemmcolumn(abuffer,bblk,c+NCT*LDC*2,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     if(TRANSB=='N')      load_irreg_b_c(b+(NCT*LDB+KCT)*2,bblk,LDB,EdgeN,BlkDimK,alpha);
     else if(TRANSB=='C') load_irreg_b_h(b+(KCT*LDB+NCT)*2,bblk,LDB,EdgeN,BlkDimK,alpha);
     else                 load_irreg_b_r(b+(KCT*LDB+NCT)*2,bblk,LDB,EdgeN,BlkDimK,alpha);
     gemmcolumnirregn(abuffer,bblk,c+NCT*LDC*2,BlksM,EdgeM,LDC,EdgeN);
     synproc(tid,numthreads,workprogress);
     KCT+=BlkDimK;
    }//loop BlkCtK++

  free(bblk);bblk=NULL;
 }//out of openmp region
 free(cchunks);cchunks=NULL;
 free(workprogress);workprogress=NULL;
 free(abuffer);abuffer=NULL;
}
