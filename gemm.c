/*M, N, K will vary for different matrices*/
#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long
#define uint8_t uchar
#define uint16_t ushort
#define uint32_t uint
#define uint64_t ulong
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#endif
#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define HALF_SUPPORT_AVAILABLE
#ifndef HALF_MAX
#define HALF_MAX 65504.f
#endif
#ifndef HALF_MIN
#define HALF_MIN 6.10352e-5f
#endif
#endif
#ifdef int_tp
#undef int_tp
#endif  //int_tp
#define int_tp int32_t
#ifdef uint_tp
#undef uint_tp
#endif  //uint_tp
#define uint_tp uint32_t
#ifdef int_tpc
#undef int_tpc
#endif  //int_tpc
#define int_tpc int32_t
#ifdef uint_tpc
#undef uint_tpc
#endif  //uint_tpc
#define uint_tpc uint32_t
#ifdef MItype
#undef MItype
#endif  //MItype
#define MItype float
#ifdef MItype0
#undef MItype0
#endif  //MItype0
#define MItype0 float
#ifdef MItype
#undef MItype
#endif  //MItype
#define MItype float
#ifdef MItype1
#undef MItype1
#endif  //MItype1
#define MItype1 float
#ifdef MItype2
#undef MItype2
#endif  //MItype2
#define MItype2 float2
#ifdef MItype4
#undef MItype4
#endif  //MItype4
#define MItype4 float4
#ifdef MItype8
#undef MItype8
#endif  //MItype8
#define MItype8 float8
#ifdef MItype16
#undef MItype16
#endif  //MItype16
#define MItype16 float16
#ifdef MOtype
#undef MOtype
#endif  //MOtype
#define MOtype float
#ifdef MOtype0
#undef MOtype0
#endif  //MOtype0
#define MOtype0 float
#ifdef MOtype
#undef MOtype
#endif  //MOtype
#define MOtype float
#ifdef MOtype1
#undef MOtype1
#endif  //MOtype1
#define MOtype1 float
#ifdef MOtype2
#undef MOtype2
#endif  //MOtype2
#define MOtype2 float2
#ifdef MOtype4
#undef MOtype4
#endif  //MOtype4
#define MOtype4 float4
#ifdef MOtype8
#undef MOtype8
#endif  //MOtype8
#define MOtype8 float8
#ifdef MOtype16
#undef MOtype16
#endif  //MOtype16
#define MOtype16 float16
#ifdef Acctype
#undef Acctype
#endif  //Acctype
#define Acctype float
#ifdef Acctype0
#undef Acctype0
#endif  //Acctype0
#define Acctype0 float
#ifdef Acctype
#undef Acctype
#endif  //Acctype
#define Acctype float
#ifdef Acctype1
#undef Acctype1
#endif  //Acctype1
#define Acctype1 float
#ifdef Acctype2
#undef Acctype2
#endif  //Acctype2
#define Acctype2 float2
#ifdef Acctype4
#undef Acctype4
#endif  //Acctype4
#define Acctype4 float4
#ifdef Acctype8
#undef Acctype8
#endif  //Acctype8
#define Acctype8 float8
#ifdef Acctype16
#undef Acctype16
#endif  //Acctype16
#define Acctype16 float16
#ifdef Difftype
#undef Difftype
#endif  //Difftype
#define Difftype float
#ifdef Difftype0
#undef Difftype0
#endif  //Difftype0
#define Difftype0 float
#ifdef Difftype
#undef Difftype
#endif  //Difftype
#define Difftype float
#ifdef Difftype1
#undef Difftype1
#endif  //Difftype1
#define Difftype1 float
#ifdef Difftype2
#undef Difftype2
#endif  //Difftype2
#define Difftype2 float2
#ifdef Difftype4
#undef Difftype4
#endif  //Difftype4
#define Difftype4 float4
#ifdef Difftype8
#undef Difftype8
#endif  //Difftype8
#define Difftype8 float8
#ifdef Difftype16
#undef Difftype16
#endif  //Difftype16
#define Difftype16 float16
#define VEC_1_0(ELEM) ELEM
#define VEC_2_0(ELEM) ELEM.x
#define VEC_2_1(ELEM) ELEM.y
#define VEC_4_0(ELEM) ELEM.x
#define VEC_4_1(ELEM) ELEM.y
#define VEC_4_2(ELEM) ELEM.z
#define VEC_4_3(ELEM) ELEM.w
#define VEC_8_0(ELEM) ELEM.s0
#define VEC_8_1(ELEM) ELEM.s1
#define VEC_8_2(ELEM) ELEM.s2
#define VEC_8_3(ELEM) ELEM.s3
#define VEC_8_4(ELEM) ELEM.s4
#define VEC_8_5(ELEM) ELEM.s5
#define VEC_8_6(ELEM) ELEM.s6
#define VEC_8_7(ELEM) ELEM.s7
#define VEC_16_0(ELEM) ELEM.s0
#define VEC_16_1(ELEM) ELEM.s1
#define VEC_16_2(ELEM) ELEM.s2
#define VEC_16_3(ELEM) ELEM.s3
#define VEC_16_4(ELEM) ELEM.s4
#define VEC_16_5(ELEM) ELEM.s5
#define VEC_16_6(ELEM) ELEM.s6
#define VEC_16_7(ELEM) ELEM.s7
#define VEC_16_8(ELEM) ELEM.s8
#define VEC_16_9(ELEM) ELEM.s9
#define VEC_16_10(ELEM) ELEM.sA
#define VEC_16_11(ELEM) ELEM.sB
#define VEC_16_12(ELEM) ELEM.sC
#define VEC_16_13(ELEM) ELEM.sD
#define VEC_16_14(ELEM) ELEM.sE
#define VEC_16_15(ELEM) ELEM.sF
#ifdef M
#undef M
#endif  //M
#define M 100
#ifdef N
#undef N
#endif  //N
#define N 500
#ifdef K
#undef K
#endif  //K
#define K 784
#ifdef v_pad_A
#undef v_pad_A
#endif  //v_pad_A
#define v_pad_A 0
#ifdef v_pad_B
#undef v_pad_B
#endif  //v_pad_B
#define v_pad_B 0
#ifdef TSM
#undef TSM
#endif  //TSM
#define TSM 128
#ifdef TSN
#undef TSN
#endif  //TSN
#define TSN 128
#ifdef TSK
#undef TSK
#endif  //TSK
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif  //TSK_UNROLL
#define TSK_UNROLL 1
#ifdef WPTM
#undef WPTM
#endif  //WPTM
#define WPTM 8
#ifdef VWM
#undef VWM
#endif  //VWM
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif  //WPTN
#define WPTN 8
#ifdef VWN
#undef VWN
#endif  //VWN
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif  //RTSM
#define RTSM 16
#ifdef RTSN
#undef RTSN
#endif  //RTSN
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif  //LPTA
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif  //LPTB
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif  //v_num_tiles
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((vec_type_hint(float4)))
__attribute__((reqd_work_group_size(16,16,1)))
void libdnn_gemm(__global const float* __restrict A_raw_ptr, const uint_tp A_offset, __global const float* __restrict B_raw_ptr, const uint_tp B_offset, __global float* __restrict C_raw_ptr, const uint_tp C_offset) {
__global const float* A = A_raw_ptr + A_offset;
__global const float* B = B_raw_ptr + B_offset;
__global float* C = C_raw_ptr + C_offset;
const int_tp tidn = get_local_id(0);
const int_tp tidm = get_local_id(1);
const int_tp offN = TSN * get_group_id(0);
const int_tp offM = TSM * get_group_id(1);
__global MItype* Cptr = C;
__local MItype Asub[128][8 + v_pad_A];
__local MItype Bsub[8][128 + v_pad_B];
{
Acctype4 Creg[WPTM][WPTN / VWN];
#pragma unroll
for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
for (int_tp wn = 0; wn < WPTN; ++wn) {
((Acctype*)(&(Creg[wm][wn / VWN])))[wn % VWN] = (Acctype)0;
}
}
{
#pragma unroll 1
for (int_tp t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int_tp la = 0; la < LPTA; ++la) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = la * RTSN * RTSM + tid;
int_tp row = id / TSK;
int_tp col = id % TSK;
int_tp tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = A[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = (MItype)0.0;
}
}
}
{
#pragma unroll 4
for (int_tp lb = 0; lb < LPTB; ++lb) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = lb * RTSN * RTSM + tid;
int_tp row = id / TSN;
int_tp col = id % TSN;
int_tp tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
Bsub[row][col] = B[(offN + col) * K + tiledIndex];
} else {
Bsub[row][col] = (MItype)0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
MItype4 Areg;
MItype4 Breg[TSK_UNROLL*WPTN/VWN];
#pragma unroll 1
for (int_tp k = 0; k < TSK; k += TSK_UNROLL) {
#pragma unroll
for (int_tp wn = 0; wn < WPTN / (VWN / TSK_UNROLL); ++wn) {
int_tp col = tidn + wn * (VWN / TSK_UNROLL) * RTSN;
VEC_4_0(Breg[wn]) = Bsub[k + 0][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k + 0][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k + 0][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k + 0][col + 48];
}
#pragma unroll
for (int_tp wm = 0; wm < WPTM / (VWM / TSK_UNROLL); ++wm) {
int_tp row = tidm + wm * (VWM / TSK_UNROLL) * RTSM;
VEC_4_0(Areg) = Asub[row + 0][k + 0];
VEC_4_1(Areg) = Asub[row + 16][k + 0];
VEC_4_2(Areg) = Asub[row + 32][k + 0];
VEC_4_3(Areg) = Asub[row + 48][k + 0];
#pragma unroll
for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
VEC_4_0(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_0(Breg[wn * TSK_UNROLL + 0])));
VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
VEC_4_1(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_1(Breg[wn * TSK_UNROLL + 0])));
VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
VEC_4_2(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_2(Breg[wn * TSK_UNROLL + 0])));
VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 0][wn]) += (Acctype)((VEC_4_0(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 1][wn]) += (Acctype)((VEC_4_1(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 2][wn]) += (Acctype)((VEC_4_2(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
VEC_4_3(Creg[wm * VWM / TSK_UNROLL + 3][wn]) += (Acctype)((VEC_4_3(Areg) * VEC_4_3(Breg[wn * TSK_UNROLL + 0])));
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int_tp wm=0; wm<WPTM; ++wm) {
int_tp globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int_tp wn=0; wn<WPTN; ++wn) {
int_tp globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = (MOtype)(((Acctype*)(&(Creg[wm][wn/VWN])))[wn%VWN]);
}
}
}
}
}
