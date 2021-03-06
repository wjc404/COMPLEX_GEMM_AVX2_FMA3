#ifdef DOUBLE
 #define IRREG_SIZE 8
 #define IRREG_VEC_TYPE __m256d
 #define IRREG_VEC_ZERO _mm256_setzero_pd
 #define IRREG_VEC_LOADA _mm256_load_pd
 #define IRREG_VEC_LOADU _mm256_loadu_pd
 #define IRREG_VEC_MASKLOAD _mm256_maskload_pd
 #define IRREG_VEC_STOREU _mm256_storeu_pd
 #define IRREG_VEC_MASKSTORE _mm256_maskstore_pd
 #define IRREG_VEC_BROAD _mm256_broadcast_sd
 #define IRREG_VEC_FMADD _mm256_fmadd_pd
 #define IRREG_VEC_FMINV _mm256_fmaddsub_pd
 #define IRREG_VEC_ADD _mm256_addsub_pd
 #define IRREG_VEC_PERM(y1) _mm256_permute_pd(y1,5)
#else
 #define IRREG_SIZE 4
 #define IRREG_VEC_TYPE __m256
 #define IRREG_VEC_ZERO _mm256_setzero_ps
 #define IRREG_VEC_LOADA _mm256_load_ps
 #define IRREG_VEC_LOADU _mm256_loadu_ps
 #define IRREG_VEC_MASKLOAD _mm256_maskload_ps
 #define IRREG_VEC_STOREU _mm256_storeu_ps
 #define IRREG_VEC_MASKSTORE _mm256_maskstore_ps
 #define IRREG_VEC_BROAD _mm256_broadcast_ss
 #define IRREG_VEC_FMADD _mm256_fmadd_ps
 #define IRREG_VEC_FMINV _mm256_fmaddsub_ps
 #define IRREG_VEC_ADD _mm256_addsub_ps
 #define IRREG_VEC_PERM(y1) _mm256_permute_ps(y1,177)
#endif

#define INIT_1col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)ctemp,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c4=IRREG_VEC_ZERO();\
}
#define INIT_3col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c4=IRREG_VEC_ZERO();cpref+=ldc*2;\
   c5=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c6=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c7=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c8=IRREG_VEC_ZERO();cpref+=ldc*2;\
   c9=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c10=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c11=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE-1),_MM_HINT_T0);\
   c12=IRREG_VEC_ZERO();\
}
#define KERNELkr {\
   c9=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c10=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c11=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c12=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c1=IRREG_VEC_FMINV(c9,b1,c1);c2=IRREG_VEC_FMINV(c10,b1,c2);c3=IRREG_VEC_FMINV(c11,b1,c3);c4=IRREG_VEC_FMINV(c12,b1,c4);\
   c9=IRREG_VEC_PERM(c9);\
   c10=IRREG_VEC_PERM(c10);\
   c11=IRREG_VEC_PERM(c11);\
   c12=IRREG_VEC_PERM(c12);\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c1=IRREG_VEC_FMINV(c9,b1,c1);c2=IRREG_VEC_FMINV(c10,b1,c2);c3=IRREG_VEC_FMINV(c11,b1,c3);c4=IRREG_VEC_FMINV(c12,b1,c4);\
}//use c9-c12 temporally
#define KERNELk1 {\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   b2=IRREG_VEC_BROAD(btemp);btemp++;\
   b3=IRREG_VEC_BROAD(btemp);btemp++;\
   a1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c1=IRREG_VEC_FMINV(a1,b1,c1);c5=IRREG_VEC_FMINV(a1,b2,c5);c9=IRREG_VEC_FMINV(a1,b3,c9);\
   a1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c2=IRREG_VEC_FMINV(a1,b1,c2);c6=IRREG_VEC_FMINV(a1,b2,c6);c10=IRREG_VEC_FMINV(a1,b3,c10);\
   a1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   c3=IRREG_VEC_FMINV(a1,b1,c3);c7=IRREG_VEC_FMINV(a1,b2,c7);c11=IRREG_VEC_FMINV(a1,b3,c11);\
   a1=IRREG_VEC_LOADA(atemp);atemp-=96/IRREG_SIZE;\
   c4=IRREG_VEC_FMINV(a1,b1,c4);c8=IRREG_VEC_FMINV(a1,b2,c8);c12=IRREG_VEC_FMINV(a1,b3,c12);\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   b2=IRREG_VEC_BROAD(btemp);btemp++;\
   b3=IRREG_VEC_BROAD(btemp);btemp++;\
   a1=IRREG_VEC_PERM(IRREG_VEC_LOADA(atemp));atemp+=32/IRREG_SIZE;\
   c1=IRREG_VEC_FMINV(a1,b1,c1);c5=IRREG_VEC_FMINV(a1,b2,c5);c9=IRREG_VEC_FMINV(a1,b3,c9);\
   a1=IRREG_VEC_PERM(IRREG_VEC_LOADA(atemp));atemp+=32/IRREG_SIZE;\
   c2=IRREG_VEC_FMINV(a1,b1,c2);c6=IRREG_VEC_FMINV(a1,b2,c6);c10=IRREG_VEC_FMINV(a1,b3,c10);\
   a1=IRREG_VEC_PERM(IRREG_VEC_LOADA(atemp));atemp+=32/IRREG_SIZE;\
   c3=IRREG_VEC_FMINV(a1,b1,c3);c7=IRREG_VEC_FMINV(a1,b2,c7);c11=IRREG_VEC_FMINV(a1,b3,c11);\
   a1=IRREG_VEC_PERM(IRREG_VEC_LOADA(atemp));atemp+=32/IRREG_SIZE;\
   c4=IRREG_VEC_FMINV(a1,b1,c4);c8=IRREG_VEC_FMINV(a1,b2,c8);c12=IRREG_VEC_FMINV(a1,b3,c12);\
}
#define KERNELk2 {\
   _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE+64)/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
   KERNELk1\
   _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE+64)/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
   KERNELk1\
}
#define STOREIRREGM_C_1col(c1,c2,c3,c4) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp,ml1),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+32/IRREG_SIZE,ml2),c2);\
   c3=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+64/IRREG_SIZE,ml3),c3);\
   c4=IRREG_VEC_ADD(IRREG_VEC_MASKLOAD(ctemp+96/IRREG_SIZE,ml4),c4);\
   IRREG_VEC_MASKSTORE(ctemp,ml1,c1);\
   IRREG_VEC_MASKSTORE(ctemp+32/IRREG_SIZE,ml2,c2);\
   IRREG_VEC_MASKSTORE(ctemp+64/IRREG_SIZE,ml3,c3);\
   IRREG_VEC_MASKSTORE(ctemp+96/IRREG_SIZE,ml4,c4);\
   ctemp+=ldc*2;\
}
#define STORE_C_1col(c1,c2,c3,c4) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+32/IRREG_SIZE),c2);\
   c3=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+64/IRREG_SIZE),c3);\
   c4=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+96/IRREG_SIZE),c4);\
   IRREG_VEC_STOREU(ctemp,c1);\
   IRREG_VEC_STOREU(ctemp+32/IRREG_SIZE,c2);\
   IRREG_VEC_STOREU(ctemp+64/IRREG_SIZE,c3);\
   IRREG_VEC_STOREU(ctemp+96/IRREG_SIZE,c4);\
   ctemp+=ldc*2;\
}
static void gemmblkirregkccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int kdim){
  register IRREG_VEC_TYPE a1,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<BlkDimN;ccol+=3){//loop over cblk-columns, calculate 3 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_3col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1 //loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_1col(c1,c2,c3,c4)
   STORE_C_1col(c5,c6,c7,c8)
   STORE_C_1col(c9,c10,c11,c12)
  }
}
static void gemmblkirregnccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int ndim){
  register IRREG_VEC_TYPE a1,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<ndim-2;ccol+=3){//loop over cblk-columns, calculate 3 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_3col
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol+=4){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    KERNELk2
    KERNELk2
   }
   STORE_C_1col(c1,c2,c3,c4)
   STORE_C_1col(c5,c6,c7,c8)
   STORE_C_1col(c9,c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_1col(c1,c2,c3,c4)
  }
}
static void gemmblkirregccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int mdim,int ndim,int kdim){
  register IRREG_VEC_TYPE a1,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;__m256i ml1,ml2,ml3,ml4;
  FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
#ifdef DOUBLE
  ml1=_mm256_setr_epi32(0,-(mdim>0),0,-(mdim>0),0,-(mdim>1),0,-(mdim>1));
  ml2=_mm256_setr_epi32(0,-(mdim>2),0,-(mdim>2),0,-(mdim>3),0,-(mdim>3));
  ml3=_mm256_setr_epi32(0,-(mdim>4),0,-(mdim>4),0,-(mdim>5),0,-(mdim>5));
  ml4=_mm256_setr_epi32(0,-(mdim>6),0,-(mdim>6),0,-(mdim>7),0,-(mdim>7));
#else //single precision
  ml1=_mm256_setr_epi32(-(mdim>0),-(mdim>0),-(mdim>1),-(mdim>1),-(mdim>2),-(mdim>2),-(mdim>3),-(mdim>3));
  ml2=_mm256_setr_epi32(-(mdim>4),-(mdim>4),-(mdim>5),-(mdim>5),-(mdim>6),-(mdim>6),-(mdim>7),-(mdim>7));
  ml3=_mm256_setr_epi32(-(mdim>8),-(mdim>8),-(mdim>9),-(mdim>9),-(mdim>10),-(mdim>10),-(mdim>11),-(mdim>11));
  ml4=_mm256_setr_epi32(-(mdim>12),-(mdim>12),-(mdim>13),-(mdim>13),-(mdim>14),-(mdim>14),-(mdim>15),-(mdim>15));
#endif
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<ndim-2;ccol+=3){//loop over cblk-columns, calculate 3 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_3col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1 //loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREIRREGM_C_1col(c1,c2,c3,c4)
   STOREIRREGM_C_1col(c5,c6,c7,c8)
   STOREIRREGM_C_1col(c9,c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELkr //loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREIRREGM_C_1col(c1,c2,c3,c4)
  }
}
