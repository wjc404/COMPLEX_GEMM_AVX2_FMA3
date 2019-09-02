# define ZERO_VALUE 1.0 //for debug use (edge mask)
static void load_irreg_a_c(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
  aread=astartpos;awrite=ablk;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      *(awrite+arow*2)=*(aread+arow*2);
      *(awrite+arow*2+1)=*(aread+arow*2+1);
    }
    for(;arow<BlkDimM;arow++){
      *(awrite+arow*2)=ZERO_VALUE;
      *(awrite+arow*2+1)=ZERO_VALUE;
    }
    aread+=lda*2;awrite+=BlkDimM*2;
  }
}
static void load_irreg_a_r(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
  aread=astartpos;awrite=ablk;
  for(arow=0;arow<mdim;arow++){
    for(acol=0;acol<kdim;acol++){
      *(awrite+acol*BlkDimM*2)=*(aread+acol*2);
      *(awrite+acol*BlkDimM*2+1)=*(aread+acol*2+1);
    }
    aread+=lda*2;awrite+=2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<BlkDimM-mdim;arow++){
      *(awrite+arow*2)=ZERO_VALUE;
      *(awrite+arow*2+1)=ZERO_VALUE;
    }
    awrite+=BlkDimM*2;
  }
}
static void load_irreg_a_h(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
  aread=astartpos;awrite=ablk;
  for(arow=0;arow<mdim;arow++){
    for(acol=0;acol<kdim;acol++){
      *(awrite+acol*BlkDimM*2)=*(aread+acol*2);
      *(awrite+acol*BlkDimM*2+1)=-*(aread+acol*2+1);
    }
    aread+=lda*2;awrite+=2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<BlkDimM-mdim;arow++){
      *(awrite+arow*2)=ZERO_VALUE;
      *(awrite+arow*2+1)=ZERO_VALUE;
    }
    awrite+=BlkDimM*2;
  }
}
static void load_reg_a_c(FLOAT *astartpos,FLOAT *ablk,int lda){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,BlkDimK);}
static void load_reg_a_r(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow;FLOAT *ar1,*ar2,*ar3,*awrite;
  for(arow=0;arow<BlkDimM;arow+=3){
    ar1=astartpos+arow*2*lda;
    ar2=ar1+2*lda;
    ar3=ar2+2*lda;
    awrite=ablk+2*arow;
    for(acol=0;acol<BlkDimK;acol++){
      *(awrite+0)=*(ar1+acol*2);
      *(awrite+1)=*(ar1+acol*2+1);
      *(awrite+2)=*(ar2+acol*2);
      *(awrite+3)=*(ar2+acol*2+1);
      *(awrite+4)=*(ar3+acol*2);
      *(awrite+5)=*(ar3+acol*2+1);
      awrite+=BlkDimM*2;
    }
  }
}
static void load_reg_a_h(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow;FLOAT *ar1,*ar2,*ar3,*awrite;
  for(arow=0;arow<BlkDimM;arow+=3){
    ar1=astartpos+arow*2*lda;
    ar2=ar1+2*lda;
    ar3=ar2+2*lda;
    awrite=ablk+2*arow;
    for(acol=0;acol<BlkDimK;acol++){
      *(awrite+0)=*(ar1+acol*2);
      *(awrite+1)=-*(ar1+acol*2+1);
      *(awrite+2)=*(ar2+acol*2);
      *(awrite+3)=-*(ar2+acol*2+1);
      *(awrite+4)=*(ar3+acol*2);
      *(awrite+5)=-*(ar3+acol*2+1);
      awrite+=BlkDimM*2;
    }
  }
}
static void load_tail_a_c(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_c(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_tail_a_r(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_r(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_tail_a_h(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_h(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_irregk_a_c(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,kdim);}
static void load_irregk_a_r(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_r(astartpos,ablk,lda,BlkDimM,kdim);}
static void load_irregk_a_h(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_h(astartpos,ablk,lda,BlkDimM,kdim);}

#define bmult1row_complex {\
 real=(*inb1);imag=inb1[1];*(outb+0)=real*alpha[0]-imag*alpha[1];*(outb+4)=imag*alpha[0]+real*alpha[1];inb1+=2;\
 real=(*inb2);imag=inb2[1];*(outb+1)=real*alpha[0]-imag*alpha[1];*(outb+5)=imag*alpha[0]+real*alpha[1];inb2+=2;\
 real=(*inb3);imag=inb3[1];*(outb+2)=real*alpha[0]-imag*alpha[1];*(outb+6)=imag*alpha[0]+real*alpha[1];inb3+=2;\
 real=(*inb4);imag=inb4[1];*(outb+3)=real*alpha[0]-imag*alpha[1];*(outb+7)=imag*alpha[0]+real*alpha[1];inb4+=2;\
}
static void load_reg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
 FLOAT *inb1,*inb2,*inb3,*inb4,*outb;FLOAT real,imag;
 int bcol,brow;
 outb=bblk;
 inb1=bstartpos;
 inb2=inb1+ldb*2;
 inb3=inb2+ldb*2;
 inb4=inb3+ldb*2;
 for(bcol=0;bcol<BlkDimN/4;bcol++){
  for(brow=0;brow<BlkDimK/4;brow++){
   bmult1row_complex
   outb+=8;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;inb4+=ldb*2;
  inb4-=(bcol==BlkDimN/4-1)*(ldb*2*BlkDimN);
  for(;brow<2*BlkDimK/4;brow++){
   bmult1row_complex
   outb+=8;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;inb4+=ldb*2;
  inb3-=(bcol==BlkDimN/4-1)*(ldb*2*BlkDimN);
  for(;brow<3*BlkDimK/4;brow++){
   bmult1row_complex
   outb+=8;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;inb4+=ldb*2;
  inb2-=(bcol==BlkDimN/4-1)*(ldb*2*BlkDimN);
  for(;brow<BlkDimK;brow++){
   bmult1row_complex
   outb+=8;
  }
  inb1+=2*(ldb-BlkDimK);
  inb2+=2*(ldb-BlkDimK);
  inb3+=2*(ldb-BlkDimK);
  inb4+=2*(ldb-BlkDimK);
 }
}
#define bmult1col_complex_retain {\
  bout[0] =bin1[0]*alpha[0]-bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]+bin1[1]*alpha[0];\
  bin1+=2;\
  bout[8] =bin2[0]*alpha[0]-bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]+bin2[1]*alpha[0];\
  bin2+=2;\
  bout[16]=bin3[0]*alpha[0]-bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]+bin3[1]*alpha[0];\
  bin3+=2;\
  bout[24]=bin4[0]*alpha[0]-bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]+bin4[1]*alpha[0];\
  bin4+=2;\
}
#define bmult2col_complex_retain {\
  bout[0] =bin1[0]*alpha[0]-bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]+bin1[1]*alpha[0];\
  bout[1] =bin1[2]*alpha[0]-bin1[3]*alpha[1];bout[5] =bin1[2]*alpha[1]+bin1[3]*alpha[0];\
  bin1+=4;\
  bout[8] =bin2[0]*alpha[0]-bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]+bin2[1]*alpha[0];\
  bout[9] =bin2[2]*alpha[0]-bin2[3]*alpha[1];bout[13]=bin2[2]*alpha[1]+bin2[3]*alpha[0];\
  bin2+=4;\
  bout[16]=bin3[0]*alpha[0]-bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]+bin3[1]*alpha[0];\
  bout[17]=bin3[2]*alpha[0]-bin3[3]*alpha[1];bout[21]=bin3[2]*alpha[1]+bin3[3]*alpha[0];\
  bin3+=4;\
  bout[24]=bin4[0]*alpha[0]-bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]+bin4[1]*alpha[0];\
  bout[25]=bin4[2]*alpha[0]-bin4[3]*alpha[1];bout[29]=bin4[2]*alpha[1]+bin4[3]*alpha[0];\
  bin4+=4;\
}
#define bmult3col_complex_retain {\
  bout[0] =bin1[0]*alpha[0]-bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]+bin1[1]*alpha[0];\
  bout[1] =bin1[2]*alpha[0]-bin1[3]*alpha[1];bout[5] =bin1[2]*alpha[1]+bin1[3]*alpha[0];\
  bout[2] =bin1[4]*alpha[0]-bin1[5]*alpha[1];bout[6] =bin1[4]*alpha[1]+bin1[5]*alpha[0];\
  bin1+=6;\
  bout[8] =bin2[0]*alpha[0]-bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]+bin2[1]*alpha[0];\
  bout[9] =bin2[2]*alpha[0]-bin2[3]*alpha[1];bout[13]=bin2[2]*alpha[1]+bin2[3]*alpha[0];\
  bout[10]=bin2[4]*alpha[0]-bin2[5]*alpha[1];bout[14]=bin2[4]*alpha[1]+bin2[5]*alpha[0];\
  bin2+=6;\
  bout[16]=bin3[0]*alpha[0]-bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]+bin3[1]*alpha[0];\
  bout[17]=bin3[2]*alpha[0]-bin3[3]*alpha[1];bout[21]=bin3[2]*alpha[1]+bin3[3]*alpha[0];\
  bout[18]=bin3[4]*alpha[0]-bin3[5]*alpha[1];bout[22]=bin3[4]*alpha[1]+bin3[5]*alpha[0];\
  bin3+=6;\
  bout[24]=bin4[0]*alpha[0]-bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]+bin4[1]*alpha[0];\
  bout[25]=bin4[2]*alpha[0]-bin4[3]*alpha[1];bout[29]=bin4[2]*alpha[1]+bin4[3]*alpha[0];\
  bout[26]=bin4[4]*alpha[0]-bin4[5]*alpha[1];bout[30]=bin4[4]*alpha[1]+bin4[5]*alpha[0];\
  bin4+=6;\
}
#define bmult4col_complex_retain {\
  bout[0] =bin1[0]*alpha[0]-bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]+bin1[1]*alpha[0];\
  bout[1] =bin1[2]*alpha[0]-bin1[3]*alpha[1];bout[5] =bin1[2]*alpha[1]+bin1[3]*alpha[0];\
  bout[2] =bin1[4]*alpha[0]-bin1[5]*alpha[1];bout[6] =bin1[4]*alpha[1]+bin1[5]*alpha[0];\
  bout[3] =bin1[6]*alpha[0]-bin1[7]*alpha[1];bout[7] =bin1[6]*alpha[1]+bin1[7]*alpha[0];\
  bin1+=8;\
  bout[8] =bin2[0]*alpha[0]-bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]+bin2[1]*alpha[0];\
  bout[9] =bin2[2]*alpha[0]-bin2[3]*alpha[1];bout[13]=bin2[2]*alpha[1]+bin2[3]*alpha[0];\
  bout[10]=bin2[4]*alpha[0]-bin2[5]*alpha[1];bout[14]=bin2[4]*alpha[1]+bin2[5]*alpha[0];\
  bout[11]=bin2[6]*alpha[0]-bin2[7]*alpha[1];bout[15]=bin2[6]*alpha[1]+bin2[7]*alpha[0];\
  bin2+=8;\
  bout[16]=bin3[0]*alpha[0]-bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]+bin3[1]*alpha[0];\
  bout[17]=bin3[2]*alpha[0]-bin3[3]*alpha[1];bout[21]=bin3[2]*alpha[1]+bin3[3]*alpha[0];\
  bout[18]=bin3[4]*alpha[0]-bin3[5]*alpha[1];bout[22]=bin3[4]*alpha[1]+bin3[5]*alpha[0];\
  bout[19]=bin3[6]*alpha[0]-bin3[7]*alpha[1];bout[23]=bin3[6]*alpha[1]+bin3[7]*alpha[0];\
  bin3+=8;\
  bout[24]=bin4[0]*alpha[0]-bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]+bin4[1]*alpha[0];\
  bout[25]=bin4[2]*alpha[0]-bin4[3]*alpha[1];bout[29]=bin4[2]*alpha[1]+bin4[3]*alpha[0];\
  bout[26]=bin4[4]*alpha[0]-bin4[5]*alpha[1];bout[30]=bin4[4]*alpha[1]+bin4[5]*alpha[0];\
  bout[27]=bin4[6]*alpha[0]-bin4[7]*alpha[1];bout[31]=bin4[6]*alpha[1]+bin4[7]*alpha[0];\
  bin4+=8;\
}
#define bmult1col_complex_conjug {\
  bout[0] =bin1[0]*alpha[0]+bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]-bin1[1]*alpha[0];\
  bin1+=2;\
  bout[8] =bin2[0]*alpha[0]+bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]-bin2[1]*alpha[0];\
  bin2+=2;\
  bout[16]=bin3[0]*alpha[0]+bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]-bin3[1]*alpha[0];\
  bin3+=2;\
  bout[24]=bin4[0]*alpha[0]+bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]-bin4[1]*alpha[0];\
  bin4+=2;\
}
#define bmult2col_complex_conjug {\
  bout[0] =bin1[0]*alpha[0]+bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]-bin1[1]*alpha[0];\
  bout[1] =bin1[2]*alpha[0]+bin1[3]*alpha[1];bout[5] =bin1[2]*alpha[1]-bin1[3]*alpha[0];\
  bin1+=4;\
  bout[8] =bin2[0]*alpha[0]+bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]-bin2[1]*alpha[0];\
  bout[9] =bin2[2]*alpha[0]+bin2[3]*alpha[1];bout[13]=bin2[2]*alpha[1]-bin2[3]*alpha[0];\
  bin2+=4;\
  bout[16]=bin3[0]*alpha[0]+bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]-bin3[1]*alpha[0];\
  bout[17]=bin3[2]*alpha[0]+bin3[3]*alpha[1];bout[21]=bin3[2]*alpha[1]-bin3[3]*alpha[0];\
  bin3+=4;\
  bout[24]=bin4[0]*alpha[0]+bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]-bin4[1]*alpha[0];\
  bout[25]=bin4[2]*alpha[0]+bin4[3]*alpha[1];bout[29]=bin4[2]*alpha[1]-bin4[3]*alpha[0];\
  bin4+=4;\
}
#define bmult3col_complex_conjug {\
  bout[0] =bin1[0]*alpha[0]+bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]-bin1[1]*alpha[0];\
  bout[1] =bin1[2]*alpha[0]+bin1[3]*alpha[1];bout[5] =bin1[2]*alpha[1]-bin1[3]*alpha[0];\
  bout[2] =bin1[4]*alpha[0]+bin1[5]*alpha[1];bout[6] =bin1[4]*alpha[1]-bin1[5]*alpha[0];\
  bin1+=6;\
  bout[8] =bin2[0]*alpha[0]+bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]-bin2[1]*alpha[0];\
  bout[9] =bin2[2]*alpha[0]+bin2[3]*alpha[1];bout[13]=bin2[2]*alpha[1]-bin2[3]*alpha[0];\
  bout[10]=bin2[4]*alpha[0]+bin2[5]*alpha[1];bout[14]=bin2[4]*alpha[1]-bin2[5]*alpha[0];\
  bin2+=6;\
  bout[16]=bin3[0]*alpha[0]+bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]-bin3[1]*alpha[0];\
  bout[17]=bin3[2]*alpha[0]+bin3[3]*alpha[1];bout[21]=bin3[2]*alpha[1]-bin3[3]*alpha[0];\
  bout[18]=bin3[4]*alpha[0]+bin3[5]*alpha[1];bout[22]=bin3[4]*alpha[1]-bin3[5]*alpha[0];\
  bin3+=6;\
  bout[24]=bin4[0]*alpha[0]+bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]-bin4[1]*alpha[0];\
  bout[25]=bin4[2]*alpha[0]+bin4[3]*alpha[1];bout[29]=bin4[2]*alpha[1]-bin4[3]*alpha[0];\
  bout[26]=bin4[4]*alpha[0]+bin4[5]*alpha[1];bout[30]=bin4[4]*alpha[1]-bin4[5]*alpha[0];\
  bin4+=6;\
}
#define bmult4col_complex_conjug {\
  bout[0] =bin1[0]*alpha[0]+bin1[1]*alpha[1];bout[4] =bin1[0]*alpha[1]-bin1[1]*alpha[0];\
  bout[1] =bin1[2]*alpha[0]+bin1[3]*alpha[1];bout[5] =bin1[2]*alpha[1]-bin1[3]*alpha[0];\
  bout[2] =bin1[4]*alpha[0]+bin1[5]*alpha[1];bout[6] =bin1[4]*alpha[1]-bin1[5]*alpha[0];\
  bout[3] =bin1[6]*alpha[0]+bin1[7]*alpha[1];bout[7] =bin1[6]*alpha[1]-bin1[7]*alpha[0];\
  bin1+=8;\
  bout[8] =bin2[0]*alpha[0]+bin2[1]*alpha[1];bout[12]=bin2[0]*alpha[1]-bin2[1]*alpha[0];\
  bout[9] =bin2[2]*alpha[0]+bin2[3]*alpha[1];bout[13]=bin2[2]*alpha[1]-bin2[3]*alpha[0];\
  bout[10]=bin2[4]*alpha[0]+bin2[5]*alpha[1];bout[14]=bin2[4]*alpha[1]-bin2[5]*alpha[0];\
  bout[11]=bin2[6]*alpha[0]+bin2[7]*alpha[1];bout[15]=bin2[6]*alpha[1]-bin2[7]*alpha[0];\
  bin2+=8;\
  bout[16]=bin3[0]*alpha[0]+bin3[1]*alpha[1];bout[20]=bin3[0]*alpha[1]-bin3[1]*alpha[0];\
  bout[17]=bin3[2]*alpha[0]+bin3[3]*alpha[1];bout[21]=bin3[2]*alpha[1]-bin3[3]*alpha[0];\
  bout[18]=bin3[4]*alpha[0]+bin3[5]*alpha[1];bout[22]=bin3[4]*alpha[1]-bin3[5]*alpha[0];\
  bout[19]=bin3[6]*alpha[0]+bin3[7]*alpha[1];bout[23]=bin3[6]*alpha[1]-bin3[7]*alpha[0];\
  bin3+=8;\
  bout[24]=bin4[0]*alpha[0]+bin4[1]*alpha[1];bout[28]=bin4[0]*alpha[1]-bin4[1]*alpha[0];\
  bout[25]=bin4[2]*alpha[0]+bin4[3]*alpha[1];bout[29]=bin4[2]*alpha[1]-bin4[3]*alpha[0];\
  bout[26]=bin4[4]*alpha[0]+bin4[5]*alpha[1];bout[30]=bin4[4]*alpha[1]-bin4[5]*alpha[0];\
  bout[27]=bin4[6]*alpha[0]+bin4[7]*alpha[1];bout[31]=bin4[6]*alpha[1]-bin4[7]*alpha[0];\
  bin4+=8;\
}
static void load_reg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
  bin1=bstartpos;bin2=bin1+ldb*2;bin3=bin2+ldb*2;bin4=bin3+ldb*2;int bshift=2*(4*ldb-BlkDimN);
  for(brow=0;brow<BlkDimK/4;brow+=4){
    bout=bblk+brow*8;
    for(bcol=0;bcol<BlkDimN;bcol+=4){
      bmult4col_complex_retain
      bout+=8*BlkDimK;
    }
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*BlkDimK/4;brow+=4){
    bout=bblk+brow*8+(BlkDimN/4-1)*8*BlkDimK+3;
    bmult1col_complex_retain
    bout=bblk+brow*8;
    for(bcol=1;bcol<BlkDimN-3;bcol+=4){
      bmult4col_complex_retain
      bout+=8*BlkDimK;
    }
    bmult3col_complex_retain
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*BlkDimK/4;brow+=4){
    bout=bblk+brow*8+(BlkDimN/4-1)*8*BlkDimK+2;
    bmult2col_complex_retain
    bout=bblk+brow*8;
    for(bcol=2;bcol<BlkDimN-2;bcol+=4){
      bmult4col_complex_retain
      bout+=8*BlkDimK;
    }
    bmult2col_complex_retain
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<BlkDimK;brow+=4){
    bout=bblk+brow*8+(BlkDimN/4-1)*8*BlkDimK+1;
    bmult3col_complex_retain
    bout=bblk+brow*8;
    for(bcol=3;bcol<BlkDimN-1;bcol+=4){
      bmult4col_complex_retain
      bout+=8*BlkDimK;
    }
    bmult1col_complex_retain
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_reg_b_h(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
  bin1=bstartpos;bin2=bin1+ldb*2;bin3=bin2+ldb*2;bin4=bin3+ldb*2;int bshift=2*(4*ldb-BlkDimN);
  for(brow=0;brow<BlkDimK/4;brow+=4){
    bout=bblk+brow*8;
    for(bcol=0;bcol<BlkDimN;bcol+=4){
      bmult4col_complex_conjug
      bout+=8*BlkDimK;
    }
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*BlkDimK/4;brow+=4){
    bout=bblk+brow*8+(BlkDimN/4-1)*8*BlkDimK+3;
    bmult1col_complex_conjug
    bout=bblk+brow*8;
    for(bcol=1;bcol<BlkDimN-3;bcol+=4){
      bmult4col_complex_conjug
      bout+=8*BlkDimK;
    }
    bmult3col_complex_conjug
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*BlkDimK/4;brow+=4){
    bout=bblk+brow*8+(BlkDimN/4-1)*8*BlkDimK+2;
    bmult2col_complex_conjug
    bout=bblk+brow*8;
    for(bcol=2;bcol<BlkDimN-2;bcol+=4){
      bmult4col_complex_conjug
      bout+=8*BlkDimK;
    }
    bmult2col_complex_conjug
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<BlkDimK;brow+=4){
    bout=bblk+brow*8+(BlkDimN/4-1)*8*BlkDimK+1;
    bmult3col_complex_conjug
    bout=bblk+brow*8;
    for(bcol=3;bcol<BlkDimN-1;bcol+=4){
      bmult4col_complex_conjug
      bout+=8*BlkDimK;
    }
    bmult1col_complex_conjug
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_irreg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;FLOAT real,imag;
  bin1=bstartpos;bin2=bin1+2*ldb;bin3=bin2+2*ldb;bin4=bin3+2*ldb;bout=bblk;
  for(bcol=0;bcol<ndim-3;bcol+=4){
    for(brow=0;brow<kdim;brow++){
      real=*(bin1+0);imag=*(bin1+1);*(bout+0)=real*alpha[0]-imag*alpha[1];*(bout+4)=real*alpha[1]+imag*alpha[0];bin1+=2;
      real=*(bin2+0);imag=*(bin2+1);*(bout+1)=real*alpha[0]-imag*alpha[1];*(bout+5)=real*alpha[1]+imag*alpha[0];bin2+=2;
      real=*(bin3+0);imag=*(bin3+1);*(bout+2)=real*alpha[0]-imag*alpha[1];*(bout+6)=real*alpha[1]+imag*alpha[0];bin3+=2;
      real=*(bin4+0);imag=*(bin4+1);*(bout+3)=real*alpha[0]-imag*alpha[1];*(bout+7)=real*alpha[1]+imag*alpha[0];bin4+=2;
      bout+=8;
    }
    bin1+=2*(4*ldb-kdim);
    bin2+=2*(4*ldb-kdim);
    bin3+=2*(4*ldb-kdim);
    bin4+=2*(4*ldb-kdim);
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      real=*(bin1+0);imag=*(bin1+1);*(bout+0)=real*alpha[0]-imag*alpha[1];*(bout+1)=real*alpha[1]+imag*alpha[0];
      bin1+=2;bout+=2;
    }
    bin1+=2*(ldb-kdim);
  }
}
static void load_irreg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;FLOAT real,imag;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*8;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      real=*(bin+0);imag=*(bin+1);*(bout+0)=real*alpha[0]-imag*alpha[1];*(bout+4)=real*alpha[1]+imag*alpha[0];
      real=*(bin+2);imag=*(bin+3);*(bout+1)=real*alpha[0]-imag*alpha[1];*(bout+5)=real*alpha[1]+imag*alpha[0];
      real=*(bin+4);imag=*(bin+5);*(bout+2)=real*alpha[0]-imag*alpha[1];*(bout+6)=real*alpha[1]+imag*alpha[0];
      real=*(bin+6);imag=*(bin+7);*(bout+3)=real*alpha[0]-imag*alpha[1];*(bout+7)=real*alpha[1]+imag*alpha[0];
      bin+=8;bout+=8*kdim;
    }
    bout-=6*brow;
    for(;bcol<ndim;bcol++){
      real=*(bin+0);imag=*(bin+1);*(bout+0)=real*alpha[0]-imag*alpha[1];*(bout+1)=real*alpha[1]+imag*alpha[0];
      bin+=2;bout+=2*kdim;
    }
    bin+=2*(ldb-ndim);
  }
}
static void load_irreg_b_h(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;FLOAT real,imag;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*8;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      real=*(bin+0);imag=*(bin+1);*(bout+0)=real*alpha[0]+imag*alpha[1];*(bout+4)=real*alpha[1]-imag*alpha[0];
      real=*(bin+2);imag=*(bin+3);*(bout+1)=real*alpha[0]+imag*alpha[1];*(bout+5)=real*alpha[1]-imag*alpha[0];
      real=*(bin+4);imag=*(bin+5);*(bout+2)=real*alpha[0]+imag*alpha[1];*(bout+6)=real*alpha[1]-imag*alpha[0];
      real=*(bin+6);imag=*(bin+7);*(bout+3)=real*alpha[0]+imag*alpha[1];*(bout+7)=real*alpha[1]-imag*alpha[0];
      bin+=8;bout+=8*kdim;
    }
    bout-=6*brow;
    for(;bcol<ndim;bcol++){
      real=*(bin+0);imag=*(bin+1);*(bout+0)=real*alpha[0]+imag*alpha[1];*(bout+1)=real*alpha[1]-imag*alpha[0];
      bin+=2;bout+=2*kdim;
    }
    bin+=2*(ldb-ndim);
  }
}
static void cmultbeta(FLOAT * __restrict__ c,int ldc,int m,int n,FLOAT * __restrict__ beta){
  int i,j;FLOAT *C0,*C;FLOAT real,imag;
  if(beta[0]==0.0 && beta[1]==0.0) return;
  C0=c;
  for(i=0;i<n;i++){
    C=C0;
    for(j=0;j<m;j++){
      real=*C;imag=*(C+1);
      *C=real*beta[0]-imag*beta[1];
      *(C+1)=real*beta[1]+imag*beta[0];
      C+=2;
    }
    C0+=ldc*2;
  }
}
