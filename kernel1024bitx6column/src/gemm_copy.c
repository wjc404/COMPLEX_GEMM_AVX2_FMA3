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
  int acol,arow;FLOAT *ar1,*ar2,*ar3,*ar4,*awrite;
  for(arow=0;arow<BlkDimM;arow+=4){
    ar1=astartpos+arow*2*lda;
    ar2=ar1+2*lda;
    ar3=ar2+2*lda;
    ar4=ar3+2*lda;
    awrite=ablk+2*arow;
    for(acol=0;acol<BlkDimK;acol++){
      awrite[0]=*(ar1+acol*2);
      awrite[1]=*(ar1+acol*2+1);
      awrite[2]=*(ar2+acol*2);
      awrite[3]=*(ar2+acol*2+1);
      awrite[4]=*(ar3+acol*2);
      awrite[5]=*(ar3+acol*2+1);
      awrite[6]=*(ar4+acol*2);
      awrite[7]=*(ar4+acol*2+1);
      awrite+=BlkDimM*2;
    }
  }
}
static void load_reg_a_h(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow;FLOAT *ar1,*ar2,*ar3,*ar4,*awrite;
  for(arow=0;arow<BlkDimM;arow+=4){
    ar1=astartpos+arow*2*lda;
    ar2=ar1+2*lda;
    ar3=ar2+2*lda;
    ar4=ar3+2*lda;
    awrite=ablk+2*arow;
    for(acol=0;acol<BlkDimK;acol++){
      awrite[0]=*(ar1+acol*2);
      awrite[1]=-*(ar1+acol*2+1);
      awrite[2]=*(ar2+acol*2);
      awrite[3]=-*(ar2+acol*2+1);
      awrite[4]=*(ar3+acol*2);
      awrite[5]=-*(ar3+acol*2+1);
      awrite[6]=*(ar4+acol*2);
      awrite[7]=-*(ar4+acol*2+1);
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
 real=(*inb1);imag=inb1[1];outb[0]=real*Ralpha-imag*Ialpha;outb[3]=imag*Ralpha+real*Ialpha;inb1+=2;\
 real=(*inb2);imag=inb2[1];outb[1]=real*Ralpha-imag*Ialpha;outb[4]=imag*Ralpha+real*Ialpha;inb2+=2;\
 real=(*inb3);imag=inb3[1];outb[2]=real*Ralpha-imag*Ialpha;outb[5]=imag*Ralpha+real*Ialpha;inb3+=2;\
}
static void load_reg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
 FLOAT *inb1,*inb2,*inb3,*outb;FLOAT real,imag,Ralpha,Ialpha;Ralpha=alpha[0];Ialpha=alpha[1];
 int bcol,brow;
 outb=bblk;
 inb1=bstartpos;
 inb2=inb1+ldb*2;
 inb3=inb2+ldb*2;
 for(bcol=0;bcol<BlkDimN/3;bcol++){
  for(brow=0;brow<BlkDimK/3;brow++){
   bmult1row_complex
   outb+=6;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;
  inb3-=(bcol==BlkDimN/3-1)*(ldb*2*BlkDimN);
  for(;brow<2*BlkDimK/3;brow++){
   bmult1row_complex
   outb+=6;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;
  inb2-=(bcol==BlkDimN/3-1)*(ldb*2*BlkDimN);
  for(;brow<BlkDimK;brow++){
   bmult1row_complex
   outb+=6;
  }
  inb1+=2*(ldb-BlkDimK);
  inb2+=2*(ldb-BlkDimK);
  inb3+=2*(ldb-BlkDimK);
 }
}
#define bmult1col_complex_retain {\
  bout[0] =bin1[0]*Ralpha-bin1[1]*Ialpha;bout[3] =bin1[0]*Ialpha+bin1[1]*Ralpha;\
  bin1+=2;\
  bout[6] =bin2[0]*Ralpha-bin2[1]*Ialpha;bout[9] =bin2[0]*Ialpha+bin2[1]*Ralpha;\
  bin2+=2;\
  bout[12]=bin3[0]*Ralpha-bin3[1]*Ialpha;bout[15]=bin3[0]*Ialpha+bin3[1]*Ralpha;\
  bin3+=2;\
  bout[18]=bin4[0]*Ralpha-bin4[1]*Ialpha;bout[21]=bin4[0]*Ialpha+bin4[1]*Ralpha;\
  bin4+=2;\
}
#define bmult2col_complex_retain {\
  bout[0] =bin1[0]*Ralpha-bin1[1]*Ialpha;bout[3] =bin1[0]*Ialpha+bin1[1]*Ralpha;\
  bout[1] =bin1[2]*Ralpha-bin1[3]*Ialpha;bout[4] =bin1[2]*Ialpha+bin1[3]*Ralpha;\
  bin1+=4;\
  bout[6] =bin2[0]*Ralpha-bin2[1]*Ialpha;bout[9] =bin2[0]*Ialpha+bin2[1]*Ralpha;\
  bout[7] =bin2[2]*Ralpha-bin2[3]*Ialpha;bout[10]=bin2[2]*Ialpha+bin2[3]*Ralpha;\
  bin2+=4;\
  bout[12]=bin3[0]*Ralpha-bin3[1]*Ialpha;bout[15]=bin3[0]*Ialpha+bin3[1]*Ralpha;\
  bout[13]=bin3[2]*Ralpha-bin3[3]*Ialpha;bout[16]=bin3[2]*Ialpha+bin3[3]*Ralpha;\
  bin3+=4;\
  bout[18]=bin4[0]*Ralpha-bin4[1]*Ialpha;bout[21]=bin4[0]*Ialpha+bin4[1]*Ralpha;\
  bout[19]=bin4[2]*Ralpha-bin4[3]*Ialpha;bout[22]=bin4[2]*Ialpha+bin4[3]*Ralpha;\
  bin4+=4;\
}
#define bmult3col_complex_retain {\
  bout[0] =bin1[0]*Ralpha-bin1[1]*Ialpha;bout[3] =bin1[0]*Ialpha+bin1[1]*Ralpha;\
  bout[1] =bin1[2]*Ralpha-bin1[3]*Ialpha;bout[4] =bin1[2]*Ialpha+bin1[3]*Ralpha;\
  bout[2] =bin1[4]*Ralpha-bin1[5]*Ialpha;bout[5] =bin1[4]*Ialpha+bin1[5]*Ralpha;\
  bin1+=6;\
  bout[6] =bin2[0]*Ralpha-bin2[1]*Ialpha;bout[9] =bin2[0]*Ialpha+bin2[1]*Ralpha;\
  bout[7] =bin2[2]*Ralpha-bin2[3]*Ialpha;bout[10]=bin2[2]*Ialpha+bin2[3]*Ralpha;\
  bout[8] =bin2[4]*Ralpha-bin2[5]*Ialpha;bout[11]=bin2[4]*Ialpha+bin2[5]*Ralpha;\
  bin2+=6;\
  bout[12]=bin3[0]*Ralpha-bin3[1]*Ialpha;bout[15]=bin3[0]*Ialpha+bin3[1]*Ralpha;\
  bout[13]=bin3[2]*Ralpha-bin3[3]*Ialpha;bout[16]=bin3[2]*Ialpha+bin3[3]*Ralpha;\
  bout[14]=bin3[4]*Ralpha-bin3[5]*Ialpha;bout[17]=bin3[4]*Ialpha+bin3[5]*Ralpha;\
  bin3+=6;\
  bout[18]=bin4[0]*Ralpha-bin4[1]*Ialpha;bout[21]=bin4[0]*Ialpha+bin4[1]*Ralpha;\
  bout[19]=bin4[2]*Ralpha-bin4[3]*Ialpha;bout[22]=bin4[2]*Ialpha+bin4[3]*Ralpha;\
  bout[20]=bin4[4]*Ralpha-bin4[5]*Ialpha;bout[23]=bin4[4]*Ialpha+bin4[5]*Ralpha;\
  bin4+=6;\
  bout+=6*BlkDimK;\
}
#define bmult1col_complex_conjug {\
  bout[0] =bin1[0]*Ralpha+bin1[1]*Ialpha;bout[3] =bin1[0]*Ialpha-bin1[1]*Ralpha;\
  bin1+=2;\
  bout[6] =bin2[0]*Ralpha+bin2[1]*Ialpha;bout[9] =bin2[0]*Ialpha-bin2[1]*Ralpha;\
  bin2+=2;\
  bout[12]=bin3[0]*Ralpha+bin3[1]*Ialpha;bout[15]=bin3[0]*Ialpha-bin3[1]*Ralpha;\
  bin3+=2;\
  bout[18]=bin4[0]*Ralpha+bin4[1]*Ialpha;bout[21]=bin4[0]*Ialpha-bin4[1]*Ralpha;\
  bin4+=2;\
}
#define bmult2col_complex_conjug {\
  bout[0] =bin1[0]*Ralpha+bin1[1]*Ialpha;bout[3] =bin1[0]*Ialpha-bin1[1]*Ralpha;\
  bout[1] =bin1[2]*Ralpha+bin1[3]*Ialpha;bout[4] =bin1[2]*Ialpha-bin1[3]*Ralpha;\
  bin1+=4;\
  bout[6] =bin2[0]*Ralpha+bin2[1]*Ialpha;bout[9] =bin2[0]*Ialpha-bin2[1]*Ralpha;\
  bout[7] =bin2[2]*Ralpha+bin2[3]*Ialpha;bout[10]=bin2[2]*Ialpha-bin2[3]*Ralpha;\
  bin2+=4;\
  bout[12]=bin3[0]*Ralpha+bin3[1]*Ialpha;bout[15]=bin3[0]*Ialpha-bin3[1]*Ralpha;\
  bout[13]=bin3[2]*Ralpha+bin3[3]*Ialpha;bout[16]=bin3[2]*Ialpha-bin3[3]*Ralpha;\
  bin3+=4;\
  bout[18]=bin4[0]*Ralpha+bin4[1]*Ialpha;bout[21]=bin4[0]*Ialpha-bin4[1]*Ralpha;\
  bout[19]=bin4[2]*Ralpha+bin4[3]*Ialpha;bout[22]=bin4[2]*Ialpha-bin4[3]*Ralpha;\
  bin4+=4;\
}
#define bmult3col_complex_conjug {\
  bout[0] =bin1[0]*Ralpha+bin1[1]*Ialpha;bout[3] =bin1[0]*Ialpha-bin1[1]*Ralpha;\
  bout[1] =bin1[2]*Ralpha+bin1[3]*Ialpha;bout[4] =bin1[2]*Ialpha-bin1[3]*Ralpha;\
  bout[2] =bin1[4]*Ralpha+bin1[5]*Ialpha;bout[5] =bin1[4]*Ialpha-bin1[5]*Ralpha;\
  bin1+=6;\
  bout[6] =bin2[0]*Ralpha+bin2[1]*Ialpha;bout[9] =bin2[0]*Ialpha-bin2[1]*Ralpha;\
  bout[7] =bin2[2]*Ralpha+bin2[3]*Ialpha;bout[10]=bin2[2]*Ialpha-bin2[3]*Ralpha;\
  bout[8] =bin2[4]*Ralpha+bin2[5]*Ialpha;bout[11]=bin2[4]*Ialpha-bin2[5]*Ralpha;\
  bin2+=6;\
  bout[12]=bin3[0]*Ralpha+bin3[1]*Ialpha;bout[15]=bin3[0]*Ialpha-bin3[1]*Ralpha;\
  bout[13]=bin3[2]*Ralpha+bin3[3]*Ialpha;bout[16]=bin3[2]*Ialpha-bin3[3]*Ralpha;\
  bout[14]=bin3[4]*Ralpha+bin3[5]*Ialpha;bout[17]=bin3[4]*Ialpha-bin3[5]*Ralpha;\
  bin3+=6;\
  bout[18]=bin4[0]*Ralpha+bin4[1]*Ialpha;bout[21]=bin4[0]*Ialpha-bin4[1]*Ralpha;\
  bout[19]=bin4[2]*Ralpha+bin4[3]*Ialpha;bout[22]=bin4[2]*Ialpha-bin4[3]*Ralpha;\
  bout[20]=bin4[4]*Ralpha+bin4[5]*Ialpha;bout[23]=bin4[4]*Ialpha-bin4[5]*Ralpha;\
  bin4+=6;\
  bout+=6*BlkDimK;\
}
static void load_reg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;FLOAT Ralpha,Ialpha;Ralpha=alpha[0];Ialpha=alpha[1];
  bin1=bstartpos;bin2=bin1+ldb*2;bin3=bin2+ldb*2;bin4=bin3+ldb*2;int bshift=2*(4*ldb-BlkDimN);
  for(brow=0;brow<BlkDimK/3;brow+=4){
    bout=bblk+brow*6;
    for(bcol=0;bcol<BlkDimN;bcol+=3) bmult3col_complex_retain
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*BlkDimK/3;brow+=4){
    bout=bblk+brow*6+(BlkDimN/3-1)*6*BlkDimK+2;
    bmult1col_complex_retain
    bout=bblk+brow*6;
    for(bcol=1;bcol<BlkDimN-2;bcol+=3) bmult3col_complex_retain
    bmult2col_complex_retain
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<BlkDimK;brow+=4){
    bout=bblk+brow*6+(BlkDimN/3-1)*6*BlkDimK+1;
    bmult2col_complex_retain
    bout=bblk+brow*6;
    for(bcol=2;bcol<BlkDimN-1;bcol+=3) bmult3col_complex_retain
    bmult1col_complex_retain
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_reg_b_h(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;FLOAT Ralpha,Ialpha;Ralpha=alpha[0];Ialpha=alpha[1];
  bin1=bstartpos;bin2=bin1+ldb*2;bin3=bin2+ldb*2;bin4=bin3+ldb*2;int bshift=2*(4*ldb-BlkDimN);
  for(brow=0;brow<BlkDimK/3;brow+=4){
    bout=bblk+brow*6;
    for(bcol=0;bcol<BlkDimN;bcol+=3) bmult3col_complex_conjug
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*BlkDimK/3;brow+=4){
    bout=bblk+brow*6+(BlkDimN/3-1)*6*BlkDimK+2;
    bmult1col_complex_conjug
    bout=bblk+brow*6;
    for(bcol=1;bcol<BlkDimN-2;bcol+=3) bmult3col_complex_conjug
    bmult2col_complex_conjug
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<BlkDimK;brow+=4){
    bout=bblk+brow*6+(BlkDimN/3-1)*6*BlkDimK+1;
    bmult2col_complex_conjug
    bout=bblk+brow*6;
    for(bcol=2;bcol<BlkDimN-1;bcol+=3) bmult3col_complex_conjug
    bmult1col_complex_conjug
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_irreg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin1,*bin2,*bin3,*bout;int bcol,brow;FLOAT real,imag,Ralpha,Ialpha;Ralpha=alpha[0];Ialpha=alpha[1];
  bin1=bstartpos;bin2=bin1+2*ldb;bin3=bin2+2*ldb;bout=bblk;
  for(bcol=0;bcol<ndim-2;bcol+=3){
    for(brow=0;brow<kdim;brow++){
      real=*bin1;imag=bin1[1];bout[0]=real*Ralpha-imag*Ialpha;bout[3]=real*Ialpha+imag*Ralpha;bin1+=2;
      real=*bin2;imag=bin2[1];bout[1]=real*Ralpha-imag*Ialpha;bout[4]=real*Ialpha+imag*Ralpha;bin2+=2;
      real=*bin3;imag=bin3[1];bout[2]=real*Ralpha-imag*Ialpha;bout[5]=real*Ialpha+imag*Ralpha;bin3+=2;
      bout+=6;
    }
    bin1+=2*(3*ldb-kdim);
    bin2+=2*(3*ldb-kdim);
    bin3+=2*(3*ldb-kdim);
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      real=*bin1;imag=bin1[1];bout[0]=real*Ralpha-imag*Ialpha;bout[1]=real*Ialpha+imag*Ralpha;
      bin1+=2;bout+=2;
    }
    bin1+=2*(ldb-kdim);
  }
}
static void load_irreg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;FLOAT real,imag;FLOAT Ralpha,Ialpha;Ralpha=alpha[0];Ialpha=alpha[1];
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*6;
    for(bcol=0;bcol<ndim-2;bcol+=3){
      real=bin[0];imag=bin[1];bout[0]=real*Ralpha-imag*Ialpha;bout[3]=real*Ialpha+imag*Ralpha;
      real=bin[2];imag=bin[3];bout[1]=real*Ralpha-imag*Ialpha;bout[4]=real*Ialpha+imag*Ralpha;
      real=bin[4];imag=bin[5];bout[2]=real*Ralpha-imag*Ialpha;bout[5]=real*Ialpha+imag*Ralpha;
      bin+=6;bout+=6*kdim;
    }
    bout-=4*brow;
    for(;bcol<ndim;bcol++){
      real=*bin;imag=bin[1];bout[0]=real*Ralpha-imag*Ialpha;bout[1]=real*Ialpha+imag*Ralpha;
      bin+=2;bout+=2*kdim;
    }
    bin+=2*(ldb-ndim);
  }
}
static void load_irreg_b_h(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;FLOAT real,imag;FLOAT Ralpha,Ialpha;Ralpha=alpha[0];Ialpha=alpha[1];
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*6;
    for(bcol=0;bcol<ndim-2;bcol+=3){
      real=bin[0];imag=bin[1];bout[0]=real*Ralpha+imag*Ialpha;bout[3]=real*Ialpha-imag*Ralpha;
      real=bin[2];imag=bin[3];bout[1]=real*Ralpha+imag*Ialpha;bout[4]=real*Ialpha-imag*Ralpha;
      real=bin[4];imag=bin[5];bout[2]=real*Ralpha+imag*Ialpha;bout[5]=real*Ialpha-imag*Ralpha;
      bin+=6;bout+=6*kdim;
    }
    bout-=4*brow;
    for(;bcol<ndim;bcol++){
      real=*bin;imag=bin[1];bout[0]=real*Ralpha+imag*Ialpha;bout[1]=real*Ialpha-imag*Ralpha;
      bin+=2;bout+=2*kdim;
    }
    bin+=2*(ldb-ndim);
  }
}
static void cmultbeta(FLOAT * __restrict__ c,int ldc,int m,int n,FLOAT * __restrict__ beta){
  int i,j;FLOAT *C0,*C;FLOAT real,imag;FLOAT Rbeta,Ibeta;Rbeta=beta[0];Ibeta=beta[1];
  if(Rbeta==0.0 && Ibeta==0.0) return;
  C0=c;
  for(i=0;i<n;i++){
    C=C0;
    for(j=0;j<m;j++){
      real=*C;imag=*(C+1);
      *C=real*Rbeta-imag*Ibeta;
      *(C+1)=real*Ibeta+imag*Rbeta;
      C+=2;
    }
    C0+=ldc*2;
  }
}
