#define A0	%rdi //ablk pointer
#define B0	%rsi //bblk pointer
#define CL      %r14 //cload pointer
#define CS      %r15 //cstore pointer
#define LDC     %rcx //ldc * sizeof(element)
#define AL      %rax //aload pointer
#define CIP  -8(%rsp)//cstartpos
//BlkDimK=128;BlkDimN=256;BlkDimM=12 for cgemm,BlkDimM=6 for zgemm.
//A_PR_BYTE=192;B_PR_ELEM=64
#ifdef DOUBLE //double precision
 #define VEC_BROAD vbroadcastsd
 #define VEC_FMA231 vfmadd231pd
 #define VEC_FMS231 vfmaddsub231pd
 #define VEC_ADD vaddsubpd
 #define VEC_PERM vpermilpd $5,
 #define MASKMOV vmaskmovpd
 #define SIZE 8
#else //single precision
 #define VEC_BROAD vbroadcastss
 #define VEC_FMA231 vfmadd231ps
 #define VEC_FMS231 vfmaddsub231ps
 #define VEC_ADD vaddsubps
 #define VEC_PERM vpermilps $177,
 #define MASKMOV vmaskmovpd
 #define SIZE 4
#endif

#define APREFUNIT (96*BlkDimK/BlkDimN)

.macro KERNEL_2 Aoff,Boff
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm4
    VEC_FMS231 %ymm0,%ymm2,%ymm5
    VEC_FMS231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm7
    VEC_FMS231 %ymm0,%ymm2,%ymm8
    VEC_FMS231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm10
    VEC_FMS231 %ymm0,%ymm2,%ymm11
    VEC_FMS231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm13
    VEC_FMS231 %ymm0,%ymm2,%ymm14
    VEC_FMS231 %ymm0,%ymm3,%ymm15
    VEC_PERM \Aoff(A0),%ymm1
    VEC_PERM \Aoff+32(A0),%ymm2
    VEC_PERM \Aoff+64(A0),%ymm3
    VEC_BROAD \Boff+4*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm4
    VEC_FMS231 %ymm0,%ymm2,%ymm5
    VEC_FMS231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+5*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm7
    VEC_FMS231 %ymm0,%ymm2,%ymm8
    VEC_FMS231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+6*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm10
    VEC_FMS231 %ymm0,%ymm2,%ymm11
    VEC_FMS231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+7*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm13
    VEC_FMS231 %ymm0,%ymm2,%ymm14
    VEC_FMS231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_F Aoff,Boff,delta,deltb
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm4
    VEC_FMS231 %ymm0,%ymm2,%ymm5
    VEC_FMS231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm7
    VEC_FMS231 %ymm0,%ymm2,%ymm8
    VEC_FMS231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm10
    VEC_FMS231 %ymm0,%ymm2,%ymm11
    VEC_FMS231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm13
    VEC_FMS231 %ymm0,%ymm2,%ymm14
    VEC_FMS231 %ymm0,%ymm3,%ymm15
    VEC_PERM \Aoff(A0),%ymm1
    VEC_PERM \Aoff+32(A0),%ymm2
    VEC_PERM \Aoff+64(A0),%ymm3
    addq $\delta,A0
    VEC_BROAD \Boff+4*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm4
    VEC_FMS231 %ymm0,%ymm2,%ymm5
    VEC_FMS231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+5*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm7
    VEC_FMS231 %ymm0,%ymm2,%ymm8
    VEC_FMS231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+6*SIZE(B0),%ymm0
    VEC_FMS231 %ymm0,%ymm1,%ymm10
    VEC_FMS231 %ymm0,%ymm2,%ymm11
    VEC_FMS231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+7*SIZE(B0),%ymm0
    addq $\deltb,B0
    VEC_FMS231 %ymm0,%ymm1,%ymm13
    VEC_FMS231 %ymm0,%ymm2,%ymm14
    VEC_FMS231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_8
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    KERNEL_2 0,0
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 A_PR_BYTE+128(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
#endif
    KERNEL_2 96,8*SIZE
    incq %r11
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    KERNEL_2 192,16*SIZE
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+24)*SIZE(B0)
#endif
    KERNEL_F 288,24*SIZE,384,32*SIZE
.endm

.macro SHIFTYMM
    vmovaps %ymm7,%ymm4
    vmovaps %ymm8,%ymm5
    vmovaps %ymm9,%ymm6
    vmovaps %ymm10,%ymm7
    vmovaps %ymm11,%ymm8
    vmovaps %ymm12,%ymm9
    vmovaps %ymm13,%ymm10
    vmovaps %ymm14,%ymm11
    vmovaps %ymm15,%ymm12
.endm

.macro PREFm12 src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 95(\src)
.endm

.macro load_mask maskaddr,r1,r2,r3
    vmovups (\maskaddr),\r1
    vmovups 32(\maskaddr),\r2
    vmovups 64(\maskaddr),\r3
.endm

.macro CLEAR r1,r2,r3
    vpxor \r1,\r1,\r1
    vpxor \r2,\r2,\r2
    vpxor \r3,\r3,\r3
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR %ymm13,%ymm14,%ymm15
.endm

.macro STORECBLK_1col
    vmovups (CS),%ymm1
    vmovups 32(CS),%ymm2
    vmovups 64(CS),%ymm3
    VEC_ADD %ymm4,%ymm1,%ymm4
    VEC_ADD %ymm5,%ymm2,%ymm5
    VEC_ADD %ymm6,%ymm3,%ymm6
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1col_irregm StoreMask
    load_mask \StoreMask,%ymm1,%ymm2,%ymm3
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm4,%ymm0,%ymm4
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm5,%ymm0,%ymm5
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm6,%ymm0,%ymm6
    MASKMOV %ymm4,%ymm1,(CS)
    MASKMOV %ymm5,%ymm2,32(CS)
    MASKMOV %ymm6,%ymm3,64(CS)
    addq LDC,CS
.endm

.macro INIT_C_3col
    CLEAR %ymm7,%ymm8,%ymm9
    CLEAR %ymm10,%ymm11,%ymm12
    CLEAR %ymm13,%ymm14,%ymm15
.endm

.macro FIN_C_3col
    vmovups (CS),%ymm1
    vmovups 32(CS),%ymm2
    vmovups 64(CS),%ymm3
    VEC_ADD %ymm4,%ymm1,%ymm4
    VEC_ADD %ymm5,%ymm2,%ymm5
    VEC_ADD %ymm6,%ymm3,%ymm6
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
    vmovups (CS),%ymm1
    vmovups 32(CS),%ymm2
    vmovups 64(CS),%ymm3
    VEC_ADD %ymm7,%ymm1,%ymm7
    VEC_ADD %ymm8,%ymm2,%ymm8
    VEC_ADD %ymm9,%ymm3,%ymm9
    vmovups %ymm7,(CS)
    vmovups %ymm8,32(CS)
    vmovups %ymm9,64(CS)
    addq LDC,CS
    vmovups (CS),%ymm1
    vmovups 32(CS),%ymm2
    vmovups 64(CS),%ymm3
    VEC_ADD %ymm10,%ymm1,%ymm10
    VEC_ADD %ymm11,%ymm2,%ymm11
    VEC_ADD %ymm12,%ymm3,%ymm12
    vmovups %ymm10,(CS)
    vmovups %ymm11,32(CS)
    vmovups %ymm12,64(CS)
.endm

.macro FIN_C_3col_irregm
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm7,%ymm0,%ymm7
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm8,%ymm0,%ymm8
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm9,%ymm0,%ymm9
    MASKMOV %ymm7,%ymm1,(CS)
    MASKMOV %ymm8,%ymm2,32(CS)
    MASKMOV %ymm9,%ymm3,64(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm10,%ymm0,%ymm10
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm11,%ymm0,%ymm11
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm12,%ymm0,%ymm12
    MASKMOV %ymm10,%ymm1,(CS)
    MASKMOV %ymm11,%ymm2,32(CS)
    MASKMOV %ymm12,%ymm3,64(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm13,%ymm0,%ymm13
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm14,%ymm0,%ymm14
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm15,%ymm0,%ymm15
    MASKMOV %ymm13,%ymm1,(CS)
    MASKMOV %ymm14,%ymm2,32(CS)
    MASKMOV %ymm15,%ymm3,64(CS)
.endm

.macro SETMASKm//use stack to store mask integer array; r8=mdim; rax=mask_pointer
#ifdef DOUBLE //double complex
    xorq %rax,%rax
    subq %r8,%rax
    addq $5,%rax
    movq %rax,-8(%rsp)
    movq %rax,-16(%rsp)
    decq %rax
    movq %rax,-24(%rsp)
    movq %rax,-32(%rsp)
    decq %rax
    movq %rax,-40(%rsp)
    movq %rax,-48(%rsp)
    decq %rax
    movq %rax,-56(%rsp)
    movq %rax,-64(%rsp)
    decq %rax
    movq %rax,-72(%rsp)
    movq %rax,-80(%rsp)
    decq %rax
    movq %rax,-88(%rsp)
    movq %rax,-96(%rsp)
    leaq -96(%rsp),%rax
#else //single complex
    xorq %rax,%rax
    subq %r8,%rax
    addq $11,%rax
    movq %rax,-8(%rsp)
    decq %rax
    movq %rax,-16(%rsp)
    decq %rax
    movq %rax,-24(%rsp)
    decq %rax
    movq %rax,-32(%rsp)
    decq %rax
    movq %rax,-40(%rsp)
    decq %rax
    movq %rax,-48(%rsp)
    decq %rax
    movq %rax,-56(%rsp)
    decq %rax
    movq %rax,-64(%rsp)
    decq %rax
    movq %rax,-72(%rsp)
    decq %rax
    movq %rax,-80(%rsp)
    decq %rax
    movq %rax,-88(%rsp)
    decq %rax
    movq %rax,-96(%rsp)
    leaq -96(%rsp),%rax
#endif
.endm

.macro SET_LDC
#ifdef DOUBLE
    salq $4,LDC
#else
    salq $3,LDC
#endif
.endm

.section .text
//enter the function gemmblkregccc, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc
.globl gemmblkregccc
.type gemmblkregccc,@function
gemmblkregccc:

    push %r15
    push %r14
    push %r12
    movq %rdx,CIP
    movq %rdi,AL
    addq $192*BlkDimK,AL //point to (prefetch) next ablk zone of abuffer,from the end to the front;
    movslq %ecx,LDC
    SET_LDC
    movq CIP,CS

    INIT_C_3col
    movq $(65536-96*BlkDimK)*281474976710656,%r10
    xorq %r12,%r12
.Louter_gemmblkregccc:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFm12 CS
    subq $APREFUNIT,AL
    prefetcht1 (AL)
#if APREFUNIT > 64
    prefetcht1 64(AL)
#endif
#if APREFUNIT > 128
    prefetcht1 128(AL)
#endif
#if APREFUNIT > 192
    prefetcht1 192(AL)
#endif
    xorq %r11,%r11
.Linner_gemmblkregccc:
    KERNEL_8
    cmpq $BlkDimK/16,%r11
    jb .Linner_gemmblkregccc

    addq %r9,A0
    prefetcht0 (A0)
# if A_PR_BYTE > 64
    prefetcht0 64(A0)
# endif
# if A_PR_BYTE > 128
    prefetcht0 128(A0)
# endif
# if A_PR_BYTE > 192
    prefetcht0 192(A0)
# endif
# if A_PR_BYTE > 256
    prefetcht0 256(A0)
# endif
    incq %r12
    STORECBLK_1col
    rorq $16,%r10
    cmpq $BlkDimN-4,%r12
    jb .Louter_gemmblkregccc

    movq AL,%r9
    subq $APREFUNIT*4,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_gemmblkregccc_last:
    PREFm12 CS
    prefetcht1 (CL)
    prefetcht1 64(CL)
    prefetcht1 95(CL)
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_last:
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht0 (%r9)
    KERNEL_2 0,0
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 A_PR_BYTE+128(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
#endif
    prefetcht0 64(%r9)
    incq %r11
    KERNEL_F 96,8*SIZE,192,16*SIZE
    prefetcht0 128(%r9)
    addq $192,%r9
    cmpq $BlkDimK/8,%r11
    jb .Linner_gemmblkregccc_last

    incq %r12
    STORECBLK_1col
    UPDATECBLK_1col
    cmpq $BlkDimN,%r12
    jb .Louter_gemmblkregccc_last

    movq CIP,CS
    FIN_C_3col

    vzeroupper
    pop %r12
    pop %r14
    pop %r15
    retq

//enter the function gemmblktailccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim
.globl gemmblktailccc
.type gemmblktailccc,@function
gemmblktailccc:

    push %r15
    push %r14
    push %r12
    push %rdx //cstartpos
    movslq %ecx,LDC
    SET_LDC
    movslq %r8d,%r8 //mdim
    SETMASKm //generate mask integers. now rax point to the base element of mask integers, just like %2 in gemm.c
    add $8,%rsp //recover rsp so "CIP" can work normally
    movq CIP,CS
    INIT_C_3col
    xorq %r12,%r12
    movq $(96*BlkDimK*281474976710656),%r10
.Louter_tail:
    UPDATECBLK_1col
    PREFm12 CS
    xorq %r11,%r11
.Linner_tail:
    KERNEL_8
    cmpq $BlkDimK/16,%r11
    jb .Linner_tail

    STORECBLK_1col_irregm %rax
    incq %r12
    movswq %r10w,%r9
    subq %r9,A0
    ror $16,%r10
    cmpq $BlkDimN,%r12
    jb .Louter_tail

    movq CIP,CS
    FIN_C_3col_irregm
    vzeroupper
    pop %r12
    pop %r14
    pop %r15
    retq

//enter the function timedelay
.globl timedelay
.type timedelay,@function
timedelay:
    xorq %r11,%r11
.Ltimedelay:
    incq %r11
    vhaddpd %ymm0,%ymm0,%ymm0
    cmpq $2000,%r11
    jb .Ltimedelay

    vzeroupper
    retq
