const int_tp tidn = get_local_id(0);
const int_tp tidm = get_local_id(1);
const int_tp offN = TSN * get_group_id(0);
const int_tp offM = TSM * get_group_id(1);
__global MItype* Cptr = C;
__local MItype Asub[TSM][TSK + v_pad_A];
__local MItype Bsub[TSK][TSN + v_pad_B];
{
Acctype Creg[WPTM][WPTN];
int16_t assignedLocalIds[WPTM][WPTN];
uint8_t maskSub[WPTM][WPTN];
#pragma unroll
for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
for (int_tp wn = 0; wn < WPTN; ++wn) {
(Creg[wm][wn]) = (Acctype)0;
}
}

{
__local int16_t idpool[RTSN * RTSM];
__local int_tp head;
__local int_tp tail;
#pragma unroll 4
for (int_tp la = 0; la < LPTA; ++la) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = la * RTSN * RTSM + tid;
int_tp row = id / TSK;
int_tp col = id % TSK;
if ((offM + row) < M && col < K) {
#if TRANS_A == 1
Asub[row][col] = A[(offM + row) + col * M];
#else
Asub[row][col] = A[(offM + row) * K + col];
#endif
} else {
Asub[row][col] = (MItype)0.0;
}
}
#pragma unroll 4
for (int_tp lb = 0; lb < LPTB; ++lb) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = lb * RTSN * RTSM + tid;
int_tp row = id / TSN;
int_tp col = id % TSN;
if ((offN + col) < N && row < K) {
#if TRANS_B == 1
Bsub[row][col] = B[(offN + col) * K + row];
#else
Bsub[row][col] = B[(offN + col) + row * N];
#endif
} else {
Bsub[row][col] = (MItype)0.0;
}
}
#pragma unroll
for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
for (int_tp wn = 0; wn < WPTN; ++wn) {
int_tp localId = tidm * RTSN + tidn;//resumes to the native id
if (localId == (int_tp)0){
head = 0;
tail = RTSN * RTSM - 1;
}
barrier(CLK_LOCAL_MEM_FENCE);
int_tp globalRow = offM + tidm + wm * RTSM;
int_tp globalCol = offN + tidn + wn * RTSN;
int_tp newId;
int_tp slot;
if (mask[globalRow * N + globalCol]){
newId = localId;
slot = atomic_inc(&head);
} 
else {
newId = localId - RTSM * RTSN;
slot = atomic_dec(&tail);
}
idpool[slot] = newId;
barrier(CLK_LOCAL_MEM_FENCE);
localId = idpool[localId];
uint8_t isMasked = (uint8_t)0;
if (localId >= 0){
int_tp col = localId % RTSN + wn * RTSN;
int_tp row = localId / RTSN + wm * RTSM;
#pragma unroll
for (int_tp k = 0; k < TSK; ++k){
Creg[wm][wn] += (Acctype)(Asub[row][k] * Bsub[k][col]);
}
isMasked = (uint8_t)1;
}
else {
localId += RTSM * RTSN;
}
maskSub[wm][wn] = isMasked;
assignedLocalIds[wm][wn] = localId;
}
}
barrier(CLK_LOCAL_MEM_FENCE);
}

#pragma unroll 1
for (int_tp t = 1; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int_tp la = 0; la < LPTA; ++la) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = la * RTSN * RTSM + tid;
int_tp row = id / TSK;
int_tp col = id % TSK;
int_tp tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
#if TRANS_A == 1
Asub[row][col] = A[(offM + row) + tiledIndex * M];
#else
Asub[row][col] = A[(offM + row) * K + tiledIndex];
#endif
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
#if TRANS_B == 1
Bsub[row][col] = B[(offN + col) * K + tiledIndex];
#else
Bsub[row][col] = B[(offN + col) + tiledIndex * N];
#endif
} else {
Bsub[row][col] = (MItype)0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
for (int_tp wn = 0; wn < WPTN; ++wn) {
if (maskSub[wm][wn]){
int_tp row = assignedLocalIds[wm][wn] / RTSN + wm * RTSM;
int_tp col = assignedLocalIds[wm][wn] % RTSN + wn * RTSN;
#pragma unroll
for (int_tp k = 0; k < TSK; ++k) {
Creg[wm][wn] += (Acctype)((Asub[row][k] * Bsub[k][col]));
}
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
}
#pragma unroll
for (int_tp wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int_tp wn=0; wn<WPTN; ++wn) {
int_tp globalRow = offM + assignedLocalIds[wm][wn] / RTSN + wm * RTSM;
int_tp globalCol = offN + assignedLocalIds[wm][wn] % RTSN + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = (MOtype)(Creg[wm][wn]);
}
}
}
}
}
