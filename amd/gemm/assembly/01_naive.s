; v0_naive computes C[m,n] = sum_k A[m,k] * B[k,n] (see ../00_naive.py).
; One 256-thread workgroup owns a 16 x 16 output tile. Each iteration consumes
; 32 K positions. A and B are FP16, the dot products accumulate in FP32,
; and C is stored as FP16. K is contiguous in A and B, and N is contiguous in C.
;
; s registers are shared by a 64-lane wave. Each lane has its own v registers.
; EXEC controls which lanes run vector instructions. VCC carries lane predicates.
; SCC holds a scalar condition or carry. Register names are reused as values expire.
;
	.amdgcn_target "amdgpu9.42-amd-amdhsa-unknown-gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	v0_naive                        ; -- Begin function v0_naive
	.p2align	8
	.type	v0_naive,@function
v0_naive:                               ; @v0_naive
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.48:
	.file	1 "/root" "01_naive.py"
	.loc	1 9 0 prologue_end              ; 01_naive.py:9:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0           ; Load the 64-bit base address of A from the kernel arguments.
	s_load_dwordx8 s[4:11], s[0:1], 0x8          ; Load B and C pointers, then M, N, K and stride_am into s4 through s11.
	s_load_dwordx4 s[12:15], s[0:1], 0x28        ; Load stride_bn and stride_cm into s12 and s13, plus two unused argument words.
	s_waitcnt lgkmcnt(0)                         ; Wait until the scalar argument loads finish.
	s_branch .LBB0_0                             ; Enter the tile calculation with the argument registers ready.
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.49:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
.Ltmp1:
	.file	2 "/root/.venvs/triton-amd/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 43 13 is_stmt 1               ; standard.py:43:13 @[ 01_naive.py:28:17 ]
	s_add_i32 s0, s9, 15                         ; Form N + 15 for ceil(N / 16).
	.loc	2 43 12 is_stmt 0               ; standard.py:43:12 @[ 01_naive.py:28:17 ]
	s_ashr_i32 s1, s0, 31                        ; Extract the sign of N + 15 as zero or all ones.
	s_lshr_b32 s1, s1, 28                        ; Turn the sign mask into a division bias of 0 or 15.
	s_add_i32 s0, s0, s1                         ; Add the bias so signed division rounds toward zero.
	s_ashr_i32 s0, s0, 4                         ; Divide by 16 to get the number of tile columns.
.Ltmp2:
	.loc	1 29 13 is_stmt 1               ; 01_naive.py:29:13
	s_abs_i32 s1, s0                             ; Take the magnitude of the divisor (tile columns).
	v_cvt_f32_u32_e32 v1, s1                     ; Convert the divisor to FP32 to estimate its reciprocal.
	s_sub_i32 s19, 0, s1                         ; Negate the divisor for the reciprocal refinement.
	s_abs_i32 s18, s16                           ; Take the magnitude of the dividend (workgroup ID in s16).
	.loc	1 48 11                         ; 01_naive.py:48:11
	v_readfirstlane_b32 s14, v0                  ; Read the first active thread ID in this wave to extract wave-uniform bits.
	.loc	1 29 13                         ; 01_naive.py:29:13
	v_rcp_f32_e32 v1, v1                         ; Estimate 1 / divisor.
	s_xor_b32 s17, s16, s0                       ; Combine the operand signs to determine the quotient sign.
	.loc	1 73 5                          ; 01_naive.py:73:5
	s_and_b32 s15, s14, 0xc0                     ; Keep thread-ID bits 6 and 7, which identify the wave in this workgroup.
	.loc	1 29 13                         ; 01_naive.py:29:13
	s_ashr_i32 s17, s17, 31                      ; Expand the quotient sign into a mask of zero or all ones.
	v_mul_f32_e32 v1, 0x4f7ffffe, v1             ; Scale the reciprocal by a float just below 2^32.
	v_cvt_u32_f32_e32 v1, v1                     ; Truncate to an unsigned fixed-point reciprocal estimate.
	s_mov_b32 s27, 0                             ; Start the K-tile counter at zero.
	.loc	1 47 11                         ; 01_naive.py:47:11
	v_and_b32_e32 v3, 15, v0                     ; Set k0 = thread_id & 15, the lane's first K offset (0..15).
	.loc	1 29 13                         ; 01_naive.py:29:13
	v_readfirstlane_b32 s20, v1                  ; Move the uniform reciprocal estimate back to a scalar register.
	s_mul_i32 s19, s19, s20                      ; Compute the low word of -divisor * estimate for the correction.
	s_mul_hi_u32 s19, s20, s19                   ; Take the high word of estimate * correction.
	s_add_i32 s20, s20, s19                      ; Refine the fixed-point reciprocal.
	s_mul_hi_u32 s19, s18, s20                   ; Take the high word of dividend * reciprocal as a trial quotient.
	s_mul_i32 s20, s19, s1                       ; Multiply the trial quotient by the divisor.
	s_sub_i32 s18, s18, s20                      ; Subtract that product from the dividend to get the trial remainder.
	s_add_i32 s20, s19, 1                        ; Prepare quotient + 1.
	s_sub_i32 s21, s18, s1                       ; Prepare remainder - divisor.
	s_cmp_ge_u32 s18, s1                         ; Does the remainder still contain a whole divisor?
	s_cselect_b32 s19, s20, s19                  ; Select the incremented quotient if it does.
	s_cselect_b32 s18, s21, s18                  ; Select the reduced remainder under the same condition.
	s_add_i32 s20, s19, 1                        ; Prepare a second possible increment.
	s_cmp_ge_u32 s18, s1                         ; Test whether a second correction is needed.
	s_cselect_b32 s1, s20, s19                   ; Select the corrected unsigned quotient.
	s_xor_b32 s1, s1, s17                        ; Flip quotient bits if the signed result must be negative.
	s_sub_i32 s1, s1, s17                        ; Finish conditional negation to obtain pid_m, the tile row.
	.loc	1 30 13                         ; 01_naive.py:30:13
	s_mul_i32 s0, s1, s0                         ; Compute pid_m * number_of_tile_columns.
	s_sub_i32 s0, s16, s0                        ; Subtract from the workgroup ID to get pid_n, the tile column.
	.loc	1 44 12                         ; 01_naive.py:44:12
; A wave handles rows r, r+2, ..., r+14. Within a wave, four groups of 16
; lanes each handle two columns (c and c+8). Lanes in a group split the K axis.
; For example, thread 0 of tile 0 handles rows 0,2,...,14, columns 0 and 8,
; and K offsets 0 and 16. Thread 1 contributes offsets 1 and 17 to those outputs.
	s_lshl_b32 s1, s1, 4                         ; Scale pid_m by 16 to get the tile's first matrix row.
	.loc	1 44 30 is_stmt 0               ; 01_naive.py:44:30
	v_and_or_b32 v1, v0, 48, s15                 ; Combine thread-ID bits 4..5 with wave bits 6..7 for the column layout.
	s_lshr_b32 s14, s14, 7                       ; Extract thread-ID bit 7, the wave's row parity (0 or 1).
	.loc	1 44 12                         ; 01_naive.py:44:12
	s_or_b32 s20, s1, s14                        ; Set r = tile_row_start + row_parity, this wave's first matrix row.
	.loc	1 46 12 is_stmt 1               ; 01_naive.py:46:12
	s_lshl_b32 s0, s0, 4                         ; Scale pid_n by 16 to get the tile's first matrix column.
	.loc	1 46 30 is_stmt 0               ; 01_naive.py:46:30
	v_lshrrev_b32_e32 v1, 4, v1                  ; Shift thread-ID bits 4..7 down to form the column selector.
	.loc	1 44 12 is_stmt 1               ; 01_naive.py:44:12
	s_or_b32 s21, s20, 4                         ; Select matrix row r+4 for this wave.
	s_or_b32 s22, s20, 6                         ; Select matrix row r+6 for this wave.
	s_or_b32 s23, s20, 8                         ; Select matrix row r+8 for this wave.
	s_or_b32 s24, s20, 10                        ; Select matrix row r+10 for this wave.
	s_or_b32 s25, s20, 12                        ; Select matrix row r+12 for this wave.
	s_or_b32 s26, s20, 14                        ; Select matrix row r+14 for this wave.
	.loc	1 46 12                         ; 01_naive.py:46:12
	v_and_or_b32 v2, v1, 7, s0                   ; Set c = tile_col_start + (selector & 7), the first output column.
	v_or3_b32 v4, s0, v1, 8                      ; Set the second output column to c+8.
.Ltmp3:
	.loc	2 43 13                         ; standard.py:43:13 @[ 01_naive.py:55:23 ]
	s_add_i32 s0, s10, 31                        ; Form K + 31 for ceil(K / 32).
	v_mov_b32_e32 v0, 0                          ; Zero the first FP32 accumulator and supply a zero for later copies.
.Ltmp4:
	.loc	1 55 5                          ; 01_naive.py:55:5
	s_cmp_lt_i32 s0, 32                          ; Test for an empty K loop.
	v_cmp_gt_i32_e32 vcc, s9, v2                 ; Set one VCC bit per lane where the first column c is below N.
	s_cbranch_scc1 .LBB0_43                      ; Skip the loads and reduction if the loop is empty.
; %bb.1:                                ; %.lr.ph
.Ltmp5:
	.loc	2 43 12                         ; standard.py:43:12 @[ 01_naive.py:55:23 ]
	s_ashr_i32 s1, s0, 31                        ; Extract the sign mask of K + 31.
	s_lshr_b32 s1, s1, 27                        ; Turn the sign mask into a division bias of 0 or 31.
	s_add_i32 s0, s0, s1                         ; Apply the bias for signed division by 32.
	s_ashr_i32 s28, s0, 5                        ; Set the loop bound to ceil(K / 32).
.Ltmp6:
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s26, s11                       ; Compute (r+14) * stride_am in elements.
	.loc	1 50 14 is_stmt 0               ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	.loc	1 44 12 is_stmt 1               ; 01_naive.py:44:12
	s_or_b32 s14, s20, 2                         ; Select matrix row r+2 for the A loads.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	.loc	1 52 14                         ; 01_naive.py:52:14
	v_lshlrev_b32_e32 v42, 1, v3                 ; Scale k0 by two bytes per FP16 element.
	v_mov_b32_e32 v43, v0                        ; Zero the upper word of the lane's 64-bit K byte offset.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[14:15], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+14,k0].
	.loc	1 50 22 is_stmt 0               ; 01_naive.py:50:22
	s_mul_i32 s0, s25, s11                       ; Compute (r+12) * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[18:19], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+12,k0].
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s24, s11                       ; Compute (r+10) * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[22:23], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+10,k0].
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s23, s11                       ; Compute (r+8) * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[26:27], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+8,k0].
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s22, s11                       ; Compute (r+6) * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[30:31], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+6,k0].
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s21, s11                       ; Compute (r+4) * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[34:35], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+4,k0].
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s14, s11                       ; Compute (r+2) * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	v_lshl_add_u64 v[38:39], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r+2,k0].
	.loc	1 50 22                         ; 01_naive.py:50:22
	s_mul_i32 s0, s20, s11                       ; Compute r * stride_am in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the A row offset into the high word.
	.loc	1 52 22 is_stmt 1               ; 01_naive.py:52:22
	v_mul_lo_u32 v6, v4, s12                     ; Compute (c+8) * stride_bn in elements.
	v_mul_lo_u32 v10, v2, s12                    ; Compute c * stride_bn in elements.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed A row offset from FP16 elements to bytes.
	.loc	1 52 14                         ; 01_naive.py:52:14
	v_ashrrev_i32_e32 v7, 31, v6                 ; Sign-extend the second B column offset to 64 bits.
	v_ashrrev_i32_e32 v11, 31, v10               ; Sign-extend the first B column offset to 64 bits.
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_add_u32 s0, s2, s0                         ; Add the low word of the A base and set the carry.
	.loc	1 52 14                         ; 01_naive.py:52:14
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[4:5]     ; Double the element offset and add the B base to locate B[0,c+8].
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[4:5]  ; Double the element offset and add the B base to locate B[0,c].
	.loc	1 50 14                         ; 01_naive.py:50:14
	s_addc_u32 s1, s3, s1                        ; Add the A base high word and carry to finish the row address.
	.loc	1 52 14                         ; 01_naive.py:52:14
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[42:43]   ; Add the lane's byte offset to locate B[k0,c+8].
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[42:43]  ; Add the lane's byte offset to locate B[k0,c].
	.loc	1 50 14                         ; 01_naive.py:50:14
	v_lshl_add_u64 v[42:43], s[0:1], 0, v[42:43]  ; Add the lane's K byte offset to form the address of A[r,k0].
	s_cmp_lt_i32 s20, s8                         ; Test whether the wave's first matrix row is below M.
	.loc	1 47 11                         ; 01_naive.py:47:11
	v_or_b32_e32 v5, 16, v3                      ; Set k1 = k0 + 16, the lane's second offset in the K tile.
	.loc	1 52 14                         ; 01_naive.py:52:14
	v_lshl_add_u64 v[8:9], v[6:7], 0, 32         ; Add 32 bytes to reach B[k1,c+8], 16 K positions later.
	v_lshl_add_u64 v[12:13], v[10:11], 0, 32     ; Add 32 bytes to reach B[k1,c], 16 K positions later.
	.loc	1 50 14                         ; 01_naive.py:50:14
	v_lshl_add_u64 v[16:17], v[14:15], 0, 32     ; Add 32 bytes to reach A[r+14,k1], 16 K positions later.
	v_lshl_add_u64 v[20:21], v[18:19], 0, 32     ; Add 32 bytes to reach A[r+12,k1], 16 K positions later.
	v_lshl_add_u64 v[24:25], v[22:23], 0, 32     ; Add 32 bytes to reach A[r+10,k1], 16 K positions later.
	v_lshl_add_u64 v[28:29], v[26:27], 0, 32     ; Add 32 bytes to reach A[r+8,k1], 16 K positions later.
	v_lshl_add_u64 v[32:33], v[30:31], 0, 32     ; Add 32 bytes to reach A[r+6,k1], 16 K positions later.
	v_lshl_add_u64 v[36:37], v[34:35], 0, 32     ; Add 32 bytes to reach A[r+4,k1], 16 K positions later.
	v_lshl_add_u64 v[40:41], v[38:39], 0, 32     ; Add 32 bytes to reach A[r+2,k1], 16 K positions later.
	s_cselect_b64 s[4:5], -1, 0                  ; Expand the row comparison to a wave-wide mask for the A loads.
	v_mov_b32_e32 v1, v0                         ; Initialize the FP32 sum for C[r+14,c+8] to zero.
	v_mov_b32_e32 v56, v0                        ; Initialize the FP32 sum for C[r+12,c] to zero.
	v_mov_b32_e32 v57, v0                        ; Initialize the FP32 sum for C[r+12,c+8] to zero.
	v_mov_b32_e32 v54, v0                        ; Initialize the FP32 sum for C[r+10,c] to zero.
	v_mov_b32_e32 v55, v0                        ; Initialize the FP32 sum for C[r+10,c+8] to zero.
	v_mov_b32_e32 v52, v0                        ; Initialize the FP32 sum for C[r+8,c] to zero.
	v_mov_b32_e32 v53, v0                        ; Initialize the FP32 sum for C[r+8,c+8] to zero.
	v_mov_b32_e32 v50, v0                        ; Initialize the FP32 sum for C[r+6,c] to zero.
	v_mov_b32_e32 v51, v0                        ; Initialize the FP32 sum for C[r+6,c+8] to zero.
	v_mov_b32_e32 v48, v0                        ; Initialize the FP32 sum for C[r+4,c] to zero.
	v_mov_b32_e32 v49, v0                        ; Initialize the FP32 sum for C[r+4,c+8] to zero.
	v_mov_b32_e32 v46, v0                        ; Initialize the FP32 sum for C[r+2,c] to zero.
	v_mov_b32_e32 v47, v0                        ; Initialize the FP32 sum for C[r+2,c+8] to zero.
	v_mov_b32_e32 v44, v0                        ; Initialize the FP32 sum for C[r,c] to zero.
	v_mov_b32_e32 v45, v0                        ; Initialize the FP32 sum for C[r,c+8] to zero.
	v_lshl_add_u64 v[58:59], v[42:43], 0, 32     ; Add 32 bytes to reach A[r,k1], 16 K positions later.
	s_branch .LBB0_3                             ; Jump to the masked loads for K tile zero.
; The loop header below (.LBB0_3) loads the inputs first, then jumps back here.
; k0 and k1 in the loop comments are relative to the current tile's K start.
; Each lane combines its two K terms for both columns. DPP then exchanges
; partial sums with lanes at XOR distances 8, 4, 2 and 1, reducing all 32 terms.
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14 is_stmt 0                ; 01_naive.py:0:14
	s_or_b64 exec, exec, s[0:1]                  ; Restore the lanes saved before the final B load.
	.loc	1 63 51 is_stmt 1               ; 01_naive.py:63:51
	s_waitcnt vmcnt(0)                           ; Wait for the vector loads before using the A and B values.
	v_cvt_f32_f16_e32 v79, v79                   ; Convert B[k1,c+8] from FP16 to FP32.
	v_cvt_f32_f16_e32 v78, v78                   ; Convert B[k1,c] from FP16 to FP32.
	.loc	1 63 20 is_stmt 0               ; 01_naive.py:63:20
	v_cvt_f32_f16_e32 v66, v66                   ; Convert A[r+4,k1] from FP16 to FP32.
	v_cvt_f32_f16_e32 v62, v62                   ; Convert A[r,k1] from FP16 to FP32.
	v_cvt_f32_f16_e32 v68, v68                   ; Convert A[r+6,k1] from FP16 to FP32.
	.loc	1 63 51                         ; 01_naive.py:63:51
	v_cvt_f32_f16_e32 v76, v76                   ; Convert B[k0,c] from FP16 to FP32.
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_pk_mul_f32 v[84:85], v[66:67], v[78:79] op_sel_hi:[0,1]  ; Multiply v66 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v66, v74                   ; Convert A[r+12,k1] from FP16 to FP32.
	v_pk_mul_f32 v[80:81], v[62:63], v[78:79] op_sel_hi:[0,1]  ; Multiply v62 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v62, v70                   ; Convert A[r+8,k1] from FP16 to FP32.
	.loc	1 63 51                         ; 01_naive.py:63:51
	v_cvt_f32_f16_e32 v77, v77                   ; Convert B[k0,c+8] from FP16 to FP32.
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_pk_mul_f32 v[90:91], v[66:67], v[78:79] op_sel_hi:[0,1]  ; Multiply v66 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v60, v60                   ; Convert A[r,k0] from FP16 to FP32.
	v_cvt_f32_f16_e32 v66, v65                   ; Convert A[r+6,k0] from FP16 to FP32.
	v_pk_mul_f32 v[86:87], v[68:69], v[78:79] op_sel_hi:[0,1]  ; Multiply v68 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v68, v75                   ; Convert A[r+14,k1] from FP16 to FP32.
	v_pk_mul_f32 v[74:75], v[62:63], v[78:79] op_sel_hi:[0,1]  ; Multiply v62 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v62, v61                   ; Convert A[r+2,k0] from FP16 to FP32.
.Ltmp7:
	.loc	2 287 12 is_stmt 1              ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_fma_f32 v[60:61], v[60:61], v[76:77], v[80:81] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[80:81].
	v_pk_fma_f32 v[80:81], v[66:67], v[76:77], v[86:87] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[86:87].
.Ltmp8:
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_cvt_f32_f16_e32 v66, v67                   ; Convert A[r+8,k0] from FP16 to FP32.
	v_cvt_f32_f16_e32 v64, v64                   ; Convert A[r+2,k1] from FP16 to FP32.
	v_cvt_f32_f16_e32 v70, v71                   ; Convert A[r+12,k0] from FP16 to FP32.
	.loc	1 55 5                          ; 01_naive.py:55:5
	s_add_i32 s27, s27, 1                        ; Count the K tile being accumulated.
.Ltmp9:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_fma_f32 v[66:67], v[66:67], v[76:77], v[74:75] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[74:75].
.Ltmp10:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v60 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v75, v61 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp11:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[60:61], v[60:61], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp12:
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_pk_mul_f32 v[82:83], v[64:65], v[78:79] op_sel_hi:[0,1]  ; Multiply v64 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v64, v72                   ; Convert A[r+10,k1] from FP16 to FP32.
.Ltmp13:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v60 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v75, v61 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
.Ltmp14:
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_cvt_f32_f16_e32 v72, v73                   ; Convert A[r+14,k0] from FP16 to FP32.
.Ltmp15:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v74 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v75, v75 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp16:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[60:61], v[60:61], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp17:
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_pk_mul_f32 v[88:89], v[64:65], v[78:79] op_sel_hi:[0,1]  ; Multiply v64 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v64, v63                   ; Convert A[r+4,k0] from FP16 to FP32.
.Ltmp18:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v60 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v75, v61 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp19:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[60:61], v[60:61], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
	v_pk_fma_f32 v[62:63], v[62:63], v[76:77], v[82:83] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[82:83].
	v_pk_fma_f32 v[64:65], v[64:65], v[76:77], v[84:85] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[84:85].
.Ltmp20:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v60 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v75, v61 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp21:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[60:61], v[60:61], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp22:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v62 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v75, v63 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp23:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[62:63], v[62:63], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp24:
	.loc	1 63 20                         ; 01_naive.py:63:20
	v_pk_mul_f32 v[78:79], v[68:69], v[78:79] op_sel_hi:[0,1]  ; Multiply v68 (A at k1) by B[k1,c] and B[k1,c+8].
	v_cvt_f32_f16_e32 v68, v69                   ; Convert A[r+10,k0] from FP16 to FP32.
.Ltmp25:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v62 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v75, v63 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
.Ltmp26:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_fma_f32 v[70:71], v[70:71], v[76:77], v[90:91] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[90:91].
.Ltmp27:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v74 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v75, v75 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp28:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[62:63], v[62:63], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
	v_pk_fma_f32 v[68:69], v[68:69], v[76:77], v[88:89] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[88:89].
	v_pk_fma_f32 v[72:73], v[72:73], v[76:77], v[78:79] op_sel_hi:[0,1,1]  ; Add A at k0 times the two B columns to the k1 products in v[78:79].
.Ltmp29:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v62 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v75, v63 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp30:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[62:63], v[62:63], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp31:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[44:45], v[44:45], v[60:61]    ; Accumulate this K tile's two reduced sums into C[r,c] and C[r,c+8].
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[42:43], v[42:43], 0, 64     ; Advance the A[r,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp32:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v62 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v75, v63 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp33:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[62:63], v[62:63], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp34:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v64 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v75, v65 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp35:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[64:65], v[64:65], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp36:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[46:47], v[46:47], v[62:63]    ; Accumulate this K tile's two reduced sums into C[r+2,c] and C[r+2,c+8].
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[58:59], v[58:59], 0, 64     ; Advance the A[r,k1] pointer by 32 FP16 elements for the next K tile.
.Ltmp37:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v64 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v75, v65 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
.Ltmp38:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[38:39], v[38:39], 0, 64     ; Advance the A[r+2,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp39:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v74 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v75, v75 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp40:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[64:65], v[64:65], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp41:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[40:41], v[40:41], 0, 64     ; Advance the A[r+2,k1] pointer by 32 FP16 elements for the next K tile.
	v_lshl_add_u64 v[34:35], v[34:35], 0, 64     ; Advance the A[r+4,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp42:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v64 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v75, v65 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp43:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[64:65], v[64:65], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp44:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[36:37], v[36:37], 0, 64     ; Advance the A[r+4,k1] pointer by 32 FP16 elements for the next K tile.
	v_lshl_add_u64 v[30:31], v[30:31], 0, 64     ; Advance the A[r+6,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp45:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v64 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v75, v65 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp46:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[64:65], v[64:65], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp47:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v74, v80 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v75, v81 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp48:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[74:75], v[80:81], v[74:75]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp49:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[48:49], v[48:49], v[64:65]    ; Accumulate this K tile's two reduced sums into C[r+4,c] and C[r+4,c+8].
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[32:33], v[32:33], 0, 64     ; Advance the A[r+6,k1] pointer by 32 FP16 elements for the next K tile.
.Ltmp50:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v74 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v77, v75 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
.Ltmp51:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[26:27], v[26:27], 0, 64     ; Advance the A[r+8,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp52:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v76 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v77, v77 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp53:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[74:75], v[74:75], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp54:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[28:29], v[28:29], 0, 64     ; Advance the A[r+8,k1] pointer by 32 FP16 elements for the next K tile.
	v_lshl_add_u64 v[22:23], v[22:23], 0, 64     ; Advance the A[r+10,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp55:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v74 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v77, v75 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp56:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[74:75], v[74:75], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp57:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[24:25], v[24:25], 0, 64     ; Advance the A[r+10,k1] pointer by 32 FP16 elements for the next K tile.
	v_lshl_add_u64 v[18:19], v[18:19], 0, 64     ; Advance the A[r+12,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp58:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v74 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v77, v75 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp59:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[74:75], v[74:75], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp60:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v66 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v77, v67 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp61:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[66:67], v[66:67], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp62:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[50:51], v[50:51], v[74:75]    ; Accumulate this K tile's two reduced sums into C[r+6,c] and C[r+6,c+8].
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[20:21], v[20:21], 0, 64     ; Advance the A[r+12,k1] pointer by 32 FP16 elements for the next K tile.
.Ltmp63:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v66 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v77, v67 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
.Ltmp64:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[14:15], v[14:15], 0, 64     ; Advance the A[r+14,k0] pointer by 32 FP16 elements for the next K tile.
.Ltmp65:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v76 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v77, v77 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp66:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[66:67], v[66:67], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp67:
	.loc	1 65 9                          ; 01_naive.py:65:9
	v_lshl_add_u64 v[16:17], v[16:17], 0, 64     ; Advance the A[r+14,k1] pointer by 32 FP16 elements for the next K tile.
	.loc	1 66 9                          ; 01_naive.py:66:9
	v_lshl_add_u64 v[10:11], v[10:11], 0, 64     ; Advance the B[k0,c] pointer by 32 FP16 elements for the next K tile.
.Ltmp68:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v66 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v77, v67 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp69:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[66:67], v[66:67], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp70:
	.loc	1 66 9                          ; 01_naive.py:66:9
	v_lshl_add_u64 v[12:13], v[12:13], 0, 64     ; Advance the B[k1,c] pointer by 32 FP16 elements for the next K tile.
	v_lshl_add_u64 v[6:7], v[6:7], 0, 64         ; Advance the B[k0,c+8] pointer by 32 FP16 elements for the next K tile.
.Ltmp71:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v66 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v77, v67 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp72:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[66:67], v[66:67], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp73:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v68 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v77, v69 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp74:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[68:69], v[68:69], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp75:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[52:53], v[52:53], v[66:67]    ; Accumulate this K tile's two reduced sums into C[r+8,c] and C[r+8,c+8].
	.loc	1 55 5                          ; 01_naive.py:55:5
	s_cmp_lg_u32 s27, s28                        ; Set SCC if another K tile remains after this one.
.Ltmp76:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v68 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v77, v69 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
.Ltmp77:
	.loc	1 66 9                          ; 01_naive.py:66:9
	v_lshl_add_u64 v[8:9], v[8:9], 0, 64         ; Advance the B[k1,c+8] pointer by 32 FP16 elements for the next K tile.
.Ltmp78:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v76 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v77, v77 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp79:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[68:69], v[68:69], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp80:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 1                                      ; Insert 2 wait states between dependent instructions.
	v_mov_b32_dpp v76, v68 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v77, v69 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp81:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[68:69], v[68:69], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp82:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 1                                      ; Insert 2 wait states between dependent instructions.
	v_mov_b32_dpp v76, v68 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v77, v69 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp83:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[68:69], v[68:69], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp84:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v70 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v77, v71 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp85:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[70:71], v[70:71], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp86:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[54:55], v[54:55], v[68:69]    ; Accumulate this K tile's two reduced sums into C[r+10,c] and C[r+10,c+8].
.Ltmp87:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 0                                      ; Insert 1 wait state between dependent instructions.
	v_mov_b32_dpp v76, v70 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v77, v71 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	s_nop 0                                      ; Insert 1 wait state between dependent instructions.
	v_mov_b32_dpp v76, v76 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v77, v77 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp88:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[70:71], v[70:71], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp89:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 1                                      ; Insert 2 wait states between dependent instructions.
	v_mov_b32_dpp v76, v70 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v77, v71 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp90:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[70:71], v[70:71], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp91:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 1                                      ; Insert 2 wait states between dependent instructions.
	v_mov_b32_dpp v76, v70 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v77, v71 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp92:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[70:71], v[70:71], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp93:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	v_mov_b32_dpp v76, v72 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
	v_mov_b32_dpp v77, v73 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 8 within the 16-lane group.
.Ltmp94:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[72:73], v[72:73], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp95:
	.loc	1 64 9                          ; 01_naive.py:64:9
	v_pk_add_f32 v[56:57], v[56:57], v[70:71]    ; Accumulate this K tile's two reduced sums into C[r+12,c] and C[r+12,c+8].
.Ltmp96:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 0                                      ; Insert 1 wait state between dependent instructions.
	v_mov_b32_dpp v76, v72 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	v_mov_b32_dpp v77, v73 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse the eight lanes in each half-row (lane XOR 7).
	s_nop 0                                      ; Insert 1 wait state between dependent instructions.
	v_mov_b32_dpp v76, v76 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
	v_mov_b32_dpp v77, v77 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Reverse each four-lane group after the half-mirror to read lane XOR 4 overall.
.Ltmp97:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[72:73], v[72:73], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp98:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 1                                      ; Insert 2 wait states between dependent instructions.
	v_mov_b32_dpp v76, v72 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
	v_mov_b32_dpp v77, v73 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from lane XOR 2 within each four-lane group.
.Ltmp99:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[72:73], v[72:73], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp100:
	.loc	2 314 12                        ; standard.py:314:12 @[ 01_naive.py:64:16 ]
	s_nop 1                                      ; Insert 2 wait states between dependent instructions.
	v_mov_b32_dpp v76, v72 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
	v_mov_b32_dpp v77, v73 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1  ; Read the partial sum from the adjacent lane (lane XOR 1).
.Ltmp101:
	.loc	2 287 12                        ; standard.py:287:12 @[ standard.py:314:12 @[ 01_naive.py:64:16 ] ]
	v_pk_add_f32 v[72:73], v[72:73], v[76:77]    ; Add the partner lane's partial sums, one FP32 addition per column.
.Ltmp102:
	.loc	1 64 9                          ; 01_naive.py:64:9
	s_nop 0                                      ; Insert 1 wait state between dependent instructions.
	v_pk_add_f32 v[0:1], v[0:1], v[72:73]        ; Accumulate this K tile's two reduced sums into C[r+14,c] and C[r+14,c+8].
	.loc	1 55 5                          ; 01_naive.py:55:5
	s_cbranch_scc0 .LBB0_44                      ; Jump to output conversion after the last tile, otherwise fall through to the loads.
; Masked input loads. The emitted code shares the r < M test across eight rows
; and the c < N test across two columns. The two K offsets have separate tests.
; A masked-off lane keeps the zero installed before its load.
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 56 25                         ; 01_naive.py:56:25
	s_lshl_b32 s0, s27, 5                        ; Compute the current tile's starting K index (32 * tile number).
	.loc	1 56 21 is_stmt 0               ; 01_naive.py:56:21
	s_sub_i32 s11, s10, s0                       ; Compute remaining = K - tile_start.
	.loc	1 58 52 is_stmt 1               ; 01_naive.py:58:52
	v_cmp_gt_i32_e64 s[0:1], s11, v3             ; Build a lane mask for k0 < remaining.
	.loc	1 58 13 is_stmt 0               ; 01_naive.py:58:13
	v_mov_b32_e32 v62, 0                         ; Supply zero for A[r,k1] in lanes that skip its load.
	.loc	1 58 29                         ; 01_naive.py:58:29
	s_and_b64 s[14:15], s[4:5], s[0:1]           ; Combine the wave's row test with the k0 mask for A.
	v_mov_b32_e32 v60, v62                       ; Supply zero for A[r,k0] in lanes that skip its load.
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[2:3], s[14:15]          ; Save the old EXEC in s[2:3] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_5                      ; Skip this load when no lane is enabled.
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v60, v[42:43], off        ; Load the FP16 bits of A[r,k0] in enabled lanes.
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[2:3]                  ; Restore the pre-load lane mask saved in s[2:3].
	.loc	1 58 52                         ; 01_naive.py:58:52
	v_cmp_gt_i32_e64 s[2:3], s11, v5             ; Build a lane mask for k1 < remaining.
	.loc	1 58 29                         ; 01_naive.py:58:29
	s_and_b64 s[16:17], s[4:5], s[2:3]           ; Combine the wave's row test with the k1 mask for A.
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_7                      ; Skip this load when no lane is enabled.
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v62, v[58:59], off        ; Load the FP16 bits of A[r,k1] in enabled lanes.
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v64, 0                         ; Supply zero for A[r+2,k1] in lanes that skip its load.
	v_mov_b32_e32 v61, v64                       ; Supply zero for A[r+2,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_9                      ; Skip this load when no lane is enabled.
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v61, v[38:39], off        ; Load the FP16 bits of A[r+2,k0] in enabled lanes.
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_11                     ; Skip this load when no lane is enabled.
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v64, v[40:41], off        ; Load the FP16 bits of A[r+2,k1] in enabled lanes.
.LBB0_11:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v66, 0                         ; Supply zero for A[r+4,k1] in lanes that skip its load.
	v_mov_b32_e32 v63, v66                       ; Supply zero for A[r+4,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_13                     ; Skip this load when no lane is enabled.
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v63, v[34:35], off        ; Load the FP16 bits of A[r+4,k0] in enabled lanes.
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_15                     ; Skip this load when no lane is enabled.
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v66, v[36:37], off        ; Load the FP16 bits of A[r+4,k1] in enabled lanes.
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v68, 0                         ; Supply zero for A[r+6,k1] in lanes that skip its load.
	v_mov_b32_e32 v65, v68                       ; Supply zero for A[r+6,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_17                     ; Skip this load when no lane is enabled.
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v65, v[30:31], off        ; Load the FP16 bits of A[r+6,k0] in enabled lanes.
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_19                     ; Skip this load when no lane is enabled.
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v68, v[32:33], off        ; Load the FP16 bits of A[r+6,k1] in enabled lanes.
.LBB0_19:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v70, 0                         ; Supply zero for A[r+8,k1] in lanes that skip its load.
	v_mov_b32_e32 v67, v70                       ; Supply zero for A[r+8,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_21                     ; Skip this load when no lane is enabled.
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v67, v[26:27], off        ; Load the FP16 bits of A[r+8,k0] in enabled lanes.
.LBB0_21:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_23                     ; Skip this load when no lane is enabled.
; %bb.22:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v70, v[28:29], off        ; Load the FP16 bits of A[r+8,k1] in enabled lanes.
.LBB0_23:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v72, 0                         ; Supply zero for A[r+10,k1] in lanes that skip its load.
	v_mov_b32_e32 v69, v72                       ; Supply zero for A[r+10,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_25                     ; Skip this load when no lane is enabled.
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v69, v[22:23], off        ; Load the FP16 bits of A[r+10,k0] in enabled lanes.
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_27                     ; Skip this load when no lane is enabled.
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v72, v[24:25], off        ; Load the FP16 bits of A[r+10,k1] in enabled lanes.
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v74, 0                         ; Supply zero for A[r+12,k1] in lanes that skip its load.
	v_mov_b32_e32 v71, v74                       ; Supply zero for A[r+12,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_29                     ; Skip this load when no lane is enabled.
; %bb.28:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v71, v[18:19], off        ; Load the FP16 bits of A[r+12,k0] in enabled lanes.
.LBB0_29:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[18:19], s[16:17]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_31                     ; Skip this load when no lane is enabled.
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v74, v[20:21], off        ; Load the FP16 bits of A[r+12,k1] in enabled lanes.
.LBB0_31:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	v_mov_b32_e32 v75, 0                         ; Supply zero for A[r+14,k1] in lanes that skip its load.
	v_mov_b32_e32 v73, v75                       ; Supply zero for A[r+14,k0] in lanes that skip its load.
	s_and_saveexec_b64 s[18:19], s[14:15]        ; Save the old EXEC in s[18:19] and enable only lanes selected by s[14:15].
	s_cbranch_execz .LBB0_33                     ; Skip this load when no lane is enabled.
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v73, v[14:15], off        ; Load the FP16 bits of A[r+14,k0] in enabled lanes.
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[18:19]                ; Restore the pre-load lane mask saved in s[18:19].
	.loc	1 58 13                         ; 01_naive.py:58:13
	s_and_saveexec_b64 s[14:15], s[16:17]        ; Save the old EXEC in s[14:15] and enable only lanes selected by s[16:17].
	s_cbranch_execz .LBB0_35                     ; Skip this load when no lane is enabled.
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v75, v[16:17], off        ; Load the FP16 bits of A[r+14,k1] in enabled lanes.
.LBB0_35:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[14:15]                ; Restore the pre-load lane mask saved in s[14:15].
	.loc	1 60 13 is_stmt 1               ; 01_naive.py:60:13
	v_mov_b32_e32 v78, 0                         ; Supply zero for B[k1,c] in lanes that skip its load.
	.loc	1 60 29 is_stmt 0               ; 01_naive.py:60:29
	s_and_b64 s[0:1], vcc, s[0:1]                ; Combine the column test with the k0 mask for B.
	v_mov_b32_e32 v76, v78                       ; Supply zero for B[k0,c] in lanes that skip its load.
	.loc	1 60 13                         ; 01_naive.py:60:13
	s_and_saveexec_b64 s[14:15], s[0:1]          ; Save the old EXEC in s[14:15] and enable only lanes selected by s[0:1].
	s_cbranch_execz .LBB0_37                     ; Skip this load when no lane is enabled.
; %bb.36:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v76, v[10:11], off        ; Load the FP16 bits of B[k0,c] in enabled lanes.
.LBB0_37:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[14:15]                ; Restore the pre-load lane mask saved in s[14:15].
	.loc	1 60 29                         ; 01_naive.py:60:29
	s_and_b64 s[2:3], vcc, s[2:3]                ; Combine the column test with the k1 mask for B.
	.loc	1 60 13                         ; 01_naive.py:60:13
	s_and_saveexec_b64 s[14:15], s[2:3]          ; Save the old EXEC in s[14:15] and enable only lanes selected by s[2:3].
	s_cbranch_execz .LBB0_39                     ; Skip this load when no lane is enabled.
; %bb.38:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v78, v[12:13], off        ; Load the FP16 bits of B[k1,c] in enabled lanes.
.LBB0_39:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[14:15]                ; Restore the pre-load lane mask saved in s[14:15].
	.loc	1 60 13                         ; 01_naive.py:60:13
	v_mov_b32_e32 v79, 0                         ; Supply zero for B[k1,c+8] in lanes that skip its load.
	v_mov_b32_e32 v77, v79                       ; Supply zero for B[k0,c+8] in lanes that skip its load.
	s_and_saveexec_b64 s[14:15], s[0:1]          ; Save the old EXEC in s[14:15] and enable only lanes selected by s[0:1].
	s_cbranch_execz .LBB0_41                     ; Skip this load when no lane is enabled.
; %bb.40:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v77, v[6:7], off          ; Load the FP16 bits of B[k0,c+8] in enabled lanes.
.LBB0_41:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13                          ; 01_naive.py:0:13
	s_or_b64 exec, exec, s[14:15]                ; Restore the pre-load lane mask saved in s[14:15].
	.loc	1 60 13                         ; 01_naive.py:60:13
	s_and_saveexec_b64 s[0:1], s[2:3]            ; Save the old EXEC in s[0:1] and enable only lanes selected by s[2:3].
	s_cbranch_execz .LBB0_2                      ; Skip this load when no lane is enabled.
; %bb.42:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_ushort v79, v[8:9], off          ; Load the FP16 bits of B[k1,c+8] in enabled lanes.
	s_branch .LBB0_2                             ; Jump back to the arithmetic for the values just loaded.
; Empty K loop. The packed FP16 representation of two zero sums is also zero.
.LBB0_43:
	.loc	1 0 13                          ; 01_naive.py:0:13
	v_mov_b32_e32 v1, 0                          ; Zero both FP16 results in this packed output pair.
	v_mov_b32_e32 v6, 0                          ; Zero both FP16 results in this packed output pair.
	v_mov_b32_e32 v7, 0                          ; Zero both FP16 results in this packed output pair.
	v_mov_b32_e32 v8, 0                          ; Zero both FP16 results in this packed output pair.
	v_mov_b32_e32 v9, 0                          ; Zero both FP16 results in this packed output pair.
	v_mov_b32_e32 v10, 0                         ; Zero both FP16 results in this packed output pair.
	v_mov_b32_e32 v11, 0                         ; Zero both FP16 results in this packed output pair.
	s_branch .LBB0_45                            ; Skip FP32 conversion and use the zero-filled output pairs.
; Loop exit. Convert each row's two FP32 sums to FP16 and pack them for stores.
.LBB0_44:                               ; %._crit_edge.loopexit
	.loc	1 73 5 is_stmt 1                ; 01_naive.py:73:5
	v_cvt_f16_f32_e32 v1, v1                     ; Round the FP32 sum for C[r+14,c+8] to FP16.
	v_cvt_f16_f32_e32 v0, v0                     ; Round the FP32 sum for C[r+14,c] to FP16.
	v_cvt_f16_f32_e32 v5, v57                    ; Round the FP32 sum for C[r+12,c+8] to FP16.
	v_cvt_f16_f32_e32 v6, v56                    ; Round the FP32 sum for C[r+12,c] to FP16.
	v_cvt_f16_f32_e32 v7, v55                    ; Round the FP32 sum for C[r+10,c+8] to FP16.
	v_cvt_f16_f32_e32 v8, v54                    ; Round the FP32 sum for C[r+10,c] to FP16.
	v_cvt_f16_f32_e32 v9, v53                    ; Round the FP32 sum for C[r+8,c+8] to FP16.
	v_cvt_f16_f32_e32 v10, v52                   ; Round the FP32 sum for C[r+8,c] to FP16.
	v_pack_b32_f16 v0, v0, v1                    ; Pack v0 (column c) low and v1 (column c+8) high.
	v_pack_b32_f16 v1, v6, v5                    ; Pack v6 (column c) low and v5 (column c+8) high.
	v_pack_b32_f16 v6, v8, v7                    ; Pack v8 (column c) low and v7 (column c+8) high.
	v_pack_b32_f16 v7, v10, v9                   ; Pack v10 (column c) low and v9 (column c+8) high.
	v_cvt_f16_f32_e32 v5, v51                    ; Round the FP32 sum for C[r+6,c+8] to FP16.
	v_cvt_f16_f32_e32 v8, v50                    ; Round the FP32 sum for C[r+6,c] to FP16.
	v_cvt_f16_f32_e32 v9, v49                    ; Round the FP32 sum for C[r+4,c+8] to FP16.
	v_cvt_f16_f32_e32 v10, v48                   ; Round the FP32 sum for C[r+4,c] to FP16.
	v_cvt_f16_f32_e32 v11, v47                   ; Round the FP32 sum for C[r+2,c+8] to FP16.
	v_cvt_f16_f32_e32 v12, v46                   ; Round the FP32 sum for C[r+2,c] to FP16.
	v_cvt_f16_f32_e32 v13, v45                   ; Round the FP32 sum for C[r,c+8] to FP16.
	v_cvt_f16_f32_e32 v14, v44                   ; Round the FP32 sum for C[r,c] to FP16.
	v_pack_b32_f16 v8, v8, v5                    ; Pack v8 (column c) low and v5 (column c+8) high.
	v_pack_b32_f16 v9, v10, v9                   ; Pack v10 (column c) low and v9 (column c+8) high.
	v_pack_b32_f16 v10, v12, v11                 ; Pack v12 (column c) low and v11 (column c+8) high.
	v_pack_b32_f16 v11, v14, v13                 ; Pack v14 (column c) low and v13 (column c+8) high.
; Writeback. Each 16-lane group holds replicated sums after the reduction.
.LBB0_45:                               ; %._crit_edge
	.loc	1 73 28                         ; 01_naive.py:73:28
	s_cmp_lt_i32 s20, s8                         ; Test the wave's first matrix row r against M.
	s_cselect_b64 s[0:1], -1, 0                  ; Expand r < M to a 64-bit lane mask.
	.loc	1 73 52 is_stmt 0               ; 01_naive.py:73:52
	v_cmp_gt_i32_e32 vcc, s9, v2                 ; Mark lanes whose first output column c is below N.
	.loc	1 73 27                         ; 01_naive.py:73:27
	s_and_b64 s[0:1], s[0:1], vcc                ; Combine the row and column tests.
	.loc	1 73 5                          ; 01_naive.py:73:5
	v_cmp_eq_u32_e32 vcc, 0, v3                  ; Select lane 0 within each 16-lane reduction group.
	s_and_b64 s[0:1], vcc, s[0:1]                ; Keep one writer for each set of replicated dot products.
	s_and_saveexec_b64 s[2:3], s[0:1]            ; Save EXEC and enable only the selected output writers.
	s_cbranch_execz .LBB0_47                     ; Skip all stores if this wave has no selected writers.
; %bb.46:                               ; %.critedge
	.loc	1 71 22 is_stmt 1               ; 01_naive.py:71:22
	s_mul_i32 s0, s26, s13                       ; Compute (r+14) * stride_cm in elements.
	.loc	1 71 14 is_stmt 0               ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	v_ashrrev_i32_e32 v5, 31, v4                 ; Sign-extend column c+8 to 64 bits.
	v_ashrrev_i32_e32 v3, 31, v2                 ; Sign-extend column c to 64 bits.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshlrev_b64 v[4:5], 1, v[4:5]              ; Convert c+8 to a byte offset into an FP16 output row.
	v_lshlrev_b64 v[2:3], 1, v[2:3]              ; Convert c to a byte offset into an FP16 output row.
	v_lshl_add_u64 v[12:13], s[0:1], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+14,c+8].
	v_lshl_add_u64 v[14:15], s[0:1], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+14,c].
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_mul_i32 s0, s25, s13                       ; Compute (r+12) * stride_cm in elements.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshl_add_u64 v[16:17], s[0:1], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+12,c+8].
	v_lshl_add_u64 v[18:19], s[0:1], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+12,c].
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_mul_i32 s0, s24, s13                       ; Compute (r+10) * stride_cm in elements.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshl_add_u64 v[20:21], s[0:1], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+10,c+8].
	v_lshl_add_u64 v[22:23], s[0:1], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+10,c].
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_mul_i32 s0, s23, s13                       ; Compute (r+8) * stride_cm in elements.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshl_add_u64 v[24:25], s[0:1], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+8,c+8].
	v_lshl_add_u64 v[26:27], s[0:1], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+8,c].
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_mul_i32 s0, s22, s13                       ; Compute (r+6) * stride_cm in elements.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshl_add_u64 v[28:29], s[0:1], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+6,c+8].
	v_lshl_add_u64 v[30:31], s[0:1], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+6,c].
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_mul_i32 s0, s21, s13                       ; Compute (r+4) * stride_cm in elements.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[2:3], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s2, s6, s2                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s3, s7, s3                        ; Add the C base high word and carry to finish the row address.
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_lshl_b32 s1, s13, 1                        ; Compute two C row strides in elements.
	s_sub_i32 s0, s0, s1                         ; Subtract two strides from the r+4 offset to obtain the r+2 offset.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshl_add_u64 v[36:37], s[0:1], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+2,c+8].
	v_lshl_add_u64 v[38:39], s[0:1], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+2,c].
	.loc	1 71 22                         ; 01_naive.py:71:22
	s_mul_i32 s0, s20, s13                       ; Compute r * stride_cm in elements.
	.loc	1 71 14                         ; 01_naive.py:71:14
	s_ashr_i32 s1, s0, 31                        ; Sign-extend the C row offset into the high word.
	s_lshl_b64 s[0:1], s[0:1], 1                 ; Convert the signed C row offset from FP16 elements to bytes.
	s_add_u32 s0, s6, s0                         ; Add the low word of the C base and set the carry.
	s_addc_u32 s1, s7, s1                        ; Add the C base high word and carry to finish the row address.
	v_lshl_add_u64 v[34:35], s[2:3], 0, v[2:3]   ; Add the column byte offset to form the address of C[r+4,c].
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[2:3]     ; Add the column byte offset to form the address of C[r,c].
	v_lshl_add_u64 v[32:33], s[2:3], 0, v[4:5]   ; Add the column byte offset to form the address of C[r+4,c+8].
	v_lshl_add_u64 v[4:5], s[0:1], 0, v[4:5]     ; Add the column byte offset to form the address of C[r,c+8].
	.loc	1 73 5 is_stmt 1                ; 01_naive.py:73:5
	global_store_short v[2:3], v11, off          ; Store the lower FP16 half of v11 to C[r,c].
	global_store_short_d16_hi v[4:5], v11, off   ; Store the upper FP16 half of v11 to C[r,c+8].
	global_store_short v[38:39], v10, off        ; Store the lower FP16 half of v10 to C[r+2,c].
	global_store_short_d16_hi v[36:37], v10, off  ; Store the upper FP16 half of v10 to C[r+2,c+8].
	global_store_short v[34:35], v9, off         ; Store the lower FP16 half of v9 to C[r+4,c].
	global_store_short_d16_hi v[32:33], v9, off  ; Store the upper FP16 half of v9 to C[r+4,c+8].
	global_store_short v[30:31], v8, off         ; Store the lower FP16 half of v8 to C[r+6,c].
	global_store_short_d16_hi v[28:29], v8, off  ; Store the upper FP16 half of v8 to C[r+6,c+8].
	global_store_short v[26:27], v7, off         ; Store the lower FP16 half of v7 to C[r+8,c].
	global_store_short_d16_hi v[24:25], v7, off  ; Store the upper FP16 half of v7 to C[r+8,c+8].
	global_store_short v[22:23], v6, off         ; Store the lower FP16 half of v6 to C[r+10,c].
	global_store_short_d16_hi v[20:21], v6, off  ; Store the upper FP16 half of v6 to C[r+10,c+8].
	global_store_short v[18:19], v1, off         ; Store the lower FP16 half of v1 to C[r+12,c].
	global_store_short_d16_hi v[16:17], v1, off  ; Store the upper FP16 half of v1 to C[r+12,c+8].
	global_store_short v[14:15], v0, off         ; Store the lower FP16 half of v0 to C[r+14,c].
	global_store_short_d16_hi v[12:13], v0, off  ; Store the upper FP16 half of v0 to C[r+14,c+8].
.LBB0_47:                               ; %.critedge28
	.loc	1 9 1                           ; 01_naive.py:9:1
	s_endpgm                                     ; Finish this wave's execution.
.Ltmp103:
.Lfunc_end0:
	.size	v0_naive, .Lfunc_end0-v0_naive
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v0_naive
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 92
		.amdhsa_next_free_sgpr 29
		.amdhsa_accum_offset 92
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lv0_naive.num_vgpr, 92
	.set .Lv0_naive.num_agpr, 0
	.set .Lv0_naive.numbered_sgpr, 29
	.set .Lv0_naive.num_named_barrier, 0
	.set .Lv0_naive.private_seg_size, 0
	.set .Lv0_naive.uses_vcc, 1
	.set .Lv0_naive.uses_flat_scratch, 0
	.set .Lv0_naive.has_dyn_sized_stack, 0
	.set .Lv0_naive.has_recursion, 0
	.set .Lv0_naive.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3516
; TotalNumSgprs: 35
; NumVgprs: 92
; NumAgprs: 0
; TotalNumVgprs: 92
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 35
; NumVGPRsForWavesPerEU: 92
; AccumOffset: 92
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 22
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.ascii	"\341\177"                      ; DW_AT_APPLE_optimized
	.byte	25                              ; DW_FORM_flag_present
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x72 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
                                        ; DW_AT_APPLE_optimized
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x4c DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	28                              ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	55                              ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x61:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	64                              ; DW_AT_call_line
	.byte	16                              ; DW_AT_call_column
	.byte	7                               ; Abbrev [7] 0x6d:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	314                             ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"01_naive.py"                   ; string offset=7 ; 01_naive.py
.Linfo_string2:
	.asciz	"/root"                         ; string offset=19 ; /root
.Linfo_string3:
	.asciz	"v0_naive"                      ; string offset=25 ; v0_naive
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 256
    .name:           v0_naive
    .private_segment_fixed_size: 0
    .sgpr_count:     35
    .sgpr_spill_count: 0
    .symbol:         v0_naive.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     92
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu9.42-amd-amdhsa-unknown-gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
