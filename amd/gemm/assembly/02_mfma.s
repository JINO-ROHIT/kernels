	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	v0_naive                        ; -- Begin function v0_naive
	.p2align	8
	.type	v0_naive,@function
v0_naive:                               ; @v0_naive
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.166:
	.file	1 "/root" "00_naive_mfma.py"
	.loc	1 8 0 prologue_end              ; 00_naive_mfma.py:8:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.167:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
.Ltmp1:
	.file	2 "/root/.venv/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 43 13 is_stmt 1               ; standard.py:43:13 @[ 00_naive_mfma.py:27:17 ]
	s_add_i32 s0, s9, 0xff
	.loc	2 43 12 is_stmt 0               ; standard.py:43:12 @[ 00_naive_mfma.py:27:17 ]
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
.Ltmp2:
	.loc	1 29 13 is_stmt 1               ; 00_naive_mfma.py:29:13
	s_abs_i32 s1, s0
	v_mov_b32_e32 v78, v0
	v_cvt_f32_u32_e32 v0, s1
	s_sub_i32 s20, 0, s1
	s_abs_i32 s19, s16
	s_xor_b32 s15, s16, s0
	v_rcp_f32_e32 v0, v0
	s_ashr_i32 s18, s15, 31
	.loc	1 86 15                         ; 00_naive_mfma.py:86:15
	v_readfirstlane_b32 s14, v78
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_b32 s17, s14, 0xc0
	.loc	1 29 13                         ; 00_naive_mfma.py:29:13
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_mov_b32 s15, 0
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_and_or_b32 v38, v78, 63, s17
	v_mov_b32_e32 v239, 0
	.loc	1 29 13                         ; 00_naive_mfma.py:29:13
	v_mul_lo_u32 v1, s20, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v0, v0, v1
	v_mul_hi_u32 v0, s19, v0
	v_mul_lo_u32 v1, v0, s1
	v_sub_u32_e32 v1, s19, v1
	v_add_u32_e32 v2, 1, v0
	v_subrev_u32_e32 v3, s1, v1
	v_cmp_le_u32_e32 vcc, s1, v1
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v132, 0
	v_cndmask_b32_e32 v0, v0, v2, vcc
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_add_u32_e32 v2, 1, v0
	v_cmp_le_u32_e32 vcc, s1, v1
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v220, 0
	v_cndmask_b32_e32 v0, v0, v2, vcc
	v_xor_b32_e32 v0, s18, v0
	v_subrev_u32_e32 v0, s18, v0
	.loc	1 30 13                         ; 00_naive_mfma.py:30:13
	v_mul_lo_u32 v1, v0, s0
	v_sub_u32_e32 v1, s16, v1
.Ltmp3:
	.loc	2 43 13                         ; standard.py:43:13 @[ 00_naive_mfma.py:70:23 ]
	s_add_i32 s0, s10, 63
.Ltmp4:
	.loc	1 52 22                         ; 00_naive_mfma.py:52:22
	v_lshlrev_b32_e32 v22, 8, v0
	.loc	1 53 22                         ; 00_naive_mfma.py:53:22
	v_lshlrev_b32_e32 v140, 8, v1
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_cmp_lt_i32 s0, 64
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v255, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v8, 0
	s_cbranch_scc1 .LBB0_35
; %bb.1:                                ; %.lr.ph
.Ltmp5:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:70:23 ]
	s_ashr_i32 s1, s0, 31
.Ltmp6:
	.loc	1 47 15                         ; 00_naive_mfma.py:47:15
	v_and_b32_e32 v6, 7, v78
.Ltmp7:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:70:23 ]
	s_lshr_b32 s1, s1, 26
.Ltmp8:
	.loc	1 53 22                         ; 00_naive_mfma.py:53:22
	v_mul_lo_u32 v0, v140, s12
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_lshrrev_b32_e32 v7, 3, v38
	.loc	1 47 15                         ; 00_naive_mfma.py:47:15
	v_lshlrev_b32_e32 v36, 3, v6
.Ltmp9:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:70:23 ]
	s_add_i32 s0, s0, s1
.Ltmp10:
	.loc	1 53 14                         ; 00_naive_mfma.py:53:14
	v_ashrrev_i32_e32 v1, 31, v0
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v26, 0xa0, v7
.Ltmp11:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:70:23 ]
	s_ashr_i32 s16, s0, 6
.Ltmp12:
	.loc	1 53 14                         ; 00_naive_mfma.py:53:14
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[4:5], s[0:1], v26, s12, v[36:37]
	.loc	1 53 14                         ; 00_naive_mfma.py:53:14
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[0:1]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v28, 0x80, v7
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_lshl_add_u64 v[10:11], v[4:5], 1, v[2:3]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[4:5], s[0:1], v28, s12, v[36:37]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v30, 0x60, v7
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_lshl_add_u64 v[12:13], v[4:5], 1, v[2:3]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[4:5], s[0:1], v30, s12, v[36:37]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v32, 64, v7
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_lshl_add_u64 v[14:15], v[4:5], 1, v[2:3]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[4:5], s[0:1], v32, s12, v[36:37]
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v8, 0xe0, v7
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v34, 32, v7
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[0:1], s[0:1], v8, s12, v[36:37]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_lshl_add_u64 v[16:17], v[4:5], 1, v[2:3]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[4:5], s[0:1], v34, s12, v[36:37]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v1, 31, v0
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v9, 0xc0, v7
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[254:255], v[0:1], 1, v[2:3]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[0:1], s[0:1], v9, s12, v[36:37]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_lshl_add_u64 v[18:19], v[4:5], 1, v[2:3]
	.loc	1 56 17                         ; 00_naive_mfma.py:56:17
	v_mad_u64_u32 v[4:5], s[0:1], v7, s12, v[36:37]
	.loc	1 59 14                         ; 00_naive_mfma.py:59:14
	v_ashrrev_i32_e32 v1, 31, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
	v_lshl_add_u64 v[20:21], v[4:5], 1, v[2:3]
	.loc	1 52 22                         ; 00_naive_mfma.py:52:22
	v_mul_lo_u32 v2, v22, s11
	.loc	1 52 14 is_stmt 0               ; 00_naive_mfma.py:52:14
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	.loc	1 55 17 is_stmt 1               ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v8, s11, v[36:37]
	.loc	1 52 14                         ; 00_naive_mfma.py:52:14
	v_lshl_add_u64 v[2:3], s[2:3], 0, v[2:3]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_accvgpr_write_b32 a135, v22
	v_lshl_add_u64 v[22:23], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v9, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[24:25], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v26, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[26:27], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v28, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[28:29], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v30, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[30:31], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v32, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[32:33], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v34, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[34:35], v[4:5], 1, v[2:3]
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mad_u64_u32 v[4:5], s[0:1], v7, s11, v[36:37]
	.loc	1 58 14                         ; 00_naive_mfma.py:58:14
	v_ashrrev_i32_e32 v5, 31, v4
	v_accvgpr_write_b32 a138, v36
	v_lshl_add_u64 v[36:37], v[4:5], 1, v[2:3]
	v_and_b32_e32 v2, 24, v78
	v_lshlrev_b32_e32 v4, 2, v38
	v_lshlrev_b32_e32 v5, 1, v2
	s_movk_i32 s0, 0x380
	v_accvgpr_write_b32 a137, v38
	v_and_or_b32 v5, v4, s0, v5
	v_lshlrev_b32_e32 v7, 5, v78
	v_bfe_i32 v38, v78, 4, 1
	s_lshl_b32 s0, s14, 2
	v_lshlrev_b32_e32 v3, 4, v6
	v_and_b32_e32 v6, 3, v78
	v_and_b32_e32 v8, 0x180, v7
	v_and_b32_e32 v39, 0x1040, v38
	s_and_b32 s0, s0, 0x200
	v_lshl_or_b32 v2, v2, 7, v3
	v_lshlrev_b32_e32 v9, 4, v6
	v_lshrrev_b32_e32 v40, 1, v78
	v_lshl_or_b32 v6, v6, 10, v8
	v_or_b32_e32 v8, s0, v39
	v_lshlrev_b32_e32 v41, 10, v78
	s_movk_i32 s0, 0x1800
	v_xor_b32_e32 v2, v5, v2
	v_and_b32_e32 v40, 16, v40
	v_and_b32_e32 v4, 0x3e0, v4
	v_and_or_b32 v3, v41, s0, v3
	v_xor_b32_e32 v5, 0x1040, v2
	v_xor_b32_e32 v9, v9, v40
	v_xor_b32_e32 v4, v4, v3
	s_movk_i32 s0, 0x410
	v_and_b32_e32 v3, 0x1e0, v7
	v_add_u32_e32 v2, 0, v2
	v_or3_b32 v6, v8, v6, v9
	s_lshl_b32 s1, s14, 3
	v_and_or_b32 v3, v38, s0, v3
	v_accvgpr_write_b32 a139, v2
	v_add_u32_e32 v2, 0, v5
	v_xor_b32_e32 v8, 32, v6
	s_and_b32 s1, s1, 0x200
	v_xor_b32_e32 v3, v3, v40
	v_accvgpr_write_b32 a140, v2
	v_add_u32_e32 v2, 0, v6
	v_xor_b32_e32 v9, 64, v6
	v_or_b32_e32 v7, s1, v3
	v_mov_b32_e32 v3, 0
	v_accvgpr_write_b32 a141, v2
	v_add_u32_e32 v2, 0, v8
	v_accvgpr_write_b32 a142, v2
	v_add_u32_e32 v2, 0, v9
	v_mov_b32_e32 v9, v3
	v_xor_b32_e32 v39, 0x60, v6
	v_xor_b32_e32 v38, 0x820, v7
	v_xor_b32_e32 v40, 0x1040, v7
	v_xor_b32_e32 v42, 0x1860, v7
	v_accvgpr_write_b32 a143, v2
	v_add_u32_e32 v2, 0, v7
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v8, v3
	v_accvgpr_write_b32 a151, v9
	v_accvgpr_write_b32 a150, v8
	v_accvgpr_write_b32 a149, v7
	v_accvgpr_write_b32 a148, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a155, v9
	v_accvgpr_write_b32 a154, v8
	v_accvgpr_write_b32 a153, v7
	v_accvgpr_write_b32 a152, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a159, v9
	v_accvgpr_write_b32 a158, v8
	v_accvgpr_write_b32 a157, v7
	v_accvgpr_write_b32 a156, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a163, v9
	v_accvgpr_write_b32 a162, v8
	v_accvgpr_write_b32 a161, v7
	v_accvgpr_write_b32 a160, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a167, v9
	v_accvgpr_write_b32 a166, v8
	v_accvgpr_write_b32 a165, v7
	v_accvgpr_write_b32 a164, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a171, v9
	v_accvgpr_write_b32 a170, v8
	v_accvgpr_write_b32 a169, v7
	v_accvgpr_write_b32 a168, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a175, v9
	v_accvgpr_write_b32 a144, v2
	v_add_u32_e32 v2, 0, v38
	v_accvgpr_write_b32 a174, v8
	v_accvgpr_write_b32 a173, v7
	v_accvgpr_write_b32 a172, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a179, v9
	v_accvgpr_write_b32 a145, v2
	v_add_u32_e32 v2, 0, v40
	v_accvgpr_write_b32 a178, v8
	v_accvgpr_write_b32 a177, v7
	v_accvgpr_write_b32 a176, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a183, v9
	v_xor_b32_e32 v41, 0x410, v4
	v_accvgpr_write_b32 a146, v2
	v_add_u32_e32 v2, 0, v42
	v_accvgpr_write_b32 a182, v8
	v_accvgpr_write_b32 a181, v7
	v_accvgpr_write_b32 a180, v6
	v_mov_b32_e32 v6, 0
	v_accvgpr_write_b32 a187, v9
	v_accvgpr_write_b32 a136, v140
	v_accvgpr_write_b32 a134, v78
	v_add_u32_e32 v43, 0, v39
	v_add_u32_e32 v44, 0, v4
	v_add_u32_e32 v45, 0, v41
	v_accvgpr_write_b32 a147, v2
	s_mov_b64 s[0:1], 0x80
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, v3
	v_mov_b32_e32 v244, v3
	v_mov_b32_e32 v245, v3
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, v3
	v_mov_b32_e32 v228, v3
	v_mov_b32_e32 v229, v3
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, v3
	v_mov_b32_e32 v248, v3
	v_mov_b32_e32 v249, v3
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, v3
	v_mov_b32_e32 v232, v3
	v_mov_b32_e32 v233, v3
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, v3
	v_mov_b32_e32 v252, v3
	v_mov_b32_e32 v253, v3
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, v3
	v_mov_b32_e32 v236, v3
	v_mov_b32_e32 v237, v3
	v_accvgpr_write_b32 a186, v8
	v_accvgpr_write_b32 a185, v7
	v_accvgpr_write_b32 a184, v6
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, v3
	v_mov_b32_e32 v48, v3
	v_mov_b32_e32 v49, v3
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, v3
	v_mov_b32_e32 v52, v3
	v_mov_b32_e32 v53, v3
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, v3
	v_mov_b32_e32 v56, v3
	v_mov_b32_e32 v57, v3
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, v3
	v_mov_b32_e32 v60, v3
	v_mov_b32_e32 v61, v3
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, v3
	v_mov_b32_e32 v64, v3
	v_mov_b32_e32 v65, v3
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v68, v3
	v_mov_b32_e32 v69, v3
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, v3
	v_mov_b32_e32 v72, v3
	v_mov_b32_e32 v73, v3
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, v3
	v_mov_b32_e32 v76, v3
	v_mov_b32_e32 v77, v3
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, v3
	v_mov_b32_e32 v80, v3
	v_mov_b32_e32 v81, v3
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v84, v3
	v_mov_b32_e32 v85, v3
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, v3
	v_mov_b32_e32 v88, v3
	v_mov_b32_e32 v89, v3
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, v3
	v_mov_b32_e32 v92, v3
	v_mov_b32_e32 v93, v3
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, v3
	v_mov_b32_e32 v96, v3
	v_mov_b32_e32 v97, v3
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, v3
	v_mov_b32_e32 v100, v3
	v_mov_b32_e32 v101, v3
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, v3
	v_mov_b32_e32 v104, v3
	v_mov_b32_e32 v105, v3
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, v3
	v_mov_b32_e32 v108, v3
	v_mov_b32_e32 v109, v3
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, v3
	v_mov_b32_e32 v112, v3
	v_mov_b32_e32 v113, v3
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, v3
	v_mov_b32_e32 v116, v3
	v_mov_b32_e32 v117, v3
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, v3
	v_mov_b32_e32 v120, v3
	v_mov_b32_e32 v121, v3
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, v3
	v_mov_b32_e32 v124, v3
	v_mov_b32_e32 v125, v3
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, v3
	v_mov_b32_e32 v128, v3
	v_mov_b32_e32 v129, v3
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, v3
	v_mov_b32_e32 v132, v3
	v_mov_b32_e32 v133, v3
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, v3
	v_mov_b32_e32 v136, v3
	v_mov_b32_e32 v137, v3
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, v3
	v_mov_b32_e32 v140, v3
	v_mov_b32_e32 v141, v3
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, v3
	v_mov_b32_e32 v144, v3
	v_mov_b32_e32 v145, v3
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, v3
	v_mov_b32_e32 v148, v3
	v_mov_b32_e32 v149, v3
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, v3
	v_mov_b32_e32 v152, v3
	v_mov_b32_e32 v153, v3
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, v3
	v_mov_b32_e32 v156, v3
	v_mov_b32_e32 v157, v3
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, v3
	v_mov_b32_e32 v160, v3
	v_mov_b32_e32 v161, v3
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, v3
	v_mov_b32_e32 v164, v3
	v_mov_b32_e32 v165, v3
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, v3
	v_mov_b32_e32 v168, v3
	v_mov_b32_e32 v169, v3
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, v3
	v_mov_b32_e32 v172, v3
	v_mov_b32_e32 v173, v3
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, v3
	v_mov_b32_e32 v176, v3
	v_mov_b32_e32 v177, v3
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, v3
	v_mov_b32_e32 v180, v3
	v_mov_b32_e32 v181, v3
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, v3
	v_mov_b32_e32 v184, v3
	v_mov_b32_e32 v185, v3
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, v3
	v_mov_b32_e32 v188, v3
	v_mov_b32_e32 v189, v3
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, v3
	v_mov_b32_e32 v192, v3
	v_mov_b32_e32 v193, v3
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, v3
	v_mov_b32_e32 v196, v3
	v_mov_b32_e32 v197, v3
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, v3
	v_mov_b32_e32 v200, v3
	v_mov_b32_e32 v201, v3
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, v3
	v_mov_b32_e32 v204, v3
	v_mov_b32_e32 v205, v3
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, v3
	v_mov_b32_e32 v208, v3
	v_mov_b32_e32 v209, v3
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, v3
	v_mov_b32_e32 v212, v3
	v_mov_b32_e32 v213, v3
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, v3
	v_mov_b32_e32 v216, v3
	v_mov_b32_e32 v217, v3
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, v3
	v_mov_b32_e32 v220, v3
	v_mov_b32_e32 v221, v3
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, v3
	v_mov_b32_e32 v224, v3
	v_mov_b32_e32 v225, v3
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, v3
	v_mov_b32_e32 v240, v3
	v_mov_b32_e32 v241, v3
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14 is_stmt 0                ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_read_b32 v38, a139
	v_accvgpr_read_b32 v39, a140
	v_accvgpr_read_b32 v40, a141
	v_accvgpr_read_b32 v41, a142
	v_accvgpr_read_b32 v42, a143
	.loc	1 73 13 is_stmt 1               ; 00_naive_mfma.py:73:13
	s_barrier
	ds_write_b128 v38, a[18:21]
	ds_write_b128 v39, a[48:51]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[118:121], v40
	ds_read_b128 a[122:125], v41
	ds_read_b128 a[126:129], v42
	ds_read_b128 a[130:133], v43
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v38, a[24:27]
	ds_write_b128 v39, a[52:55]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[78:81], v40
	ds_read_b128 a[82:85], v41
	ds_read_b128 a[86:89], v42
	ds_read_b128 a[90:93], v43
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v38, a[30:33]
	ds_write_b128 v39, a[56:59]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[46:49], v40
	ds_read_b128 a[50:53], v41
	ds_read_b128 a[54:57], v42
	ds_read_b128 a[58:61], v43
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v38, a[36:39]
	ds_write_b128 v39, a[62:65]
	v_accvgpr_read_b32 v38, a144
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[28:31], v40
	ds_read_b128 a[24:27], v41
	ds_read_b128 a[20:23], v42
	ds_read_b128 a[16:19], v43
	.loc	1 74 13                         ; 00_naive_mfma.py:74:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v44, a[42:45]
	ds_write_b128 v45, a[70:73]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[32:35], v38
	v_accvgpr_read_b32 v39, a145
	ds_read_b128 a[36:39], v39
	v_accvgpr_read_b32 v40, a146
	ds_read_b128 a[40:43], v40
	v_accvgpr_read_b32 v41, a147
	ds_read_b128 a[62:65], v41
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v44, a[12:15]
	ds_write_b128 v45, a[66:69]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[10:13], v38
	ds_read_b128 a[66:69], v39
	ds_read_b128 a[70:73], v40
	ds_read_b128 a[74:77], v41
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v44, a[6:9]
	ds_write_b128 v45, a[94:97]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[4:7], v38
	ds_read_b128 a[94:97], v39
	ds_read_b128 a[98:101], v40
	ds_read_b128 a[102:105], v41
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v44, a[0:3]
	ds_write_b128 v45, a[106:109]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[0:3], v38
	ds_read_b128 a[106:109], v39
	ds_read_b128 a[114:117], v40
	ds_read_b128 a[110:113], v41
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[38:41], a[34:35], a[78:79], v[46:49]
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_add_i32 s15, s15, 1
	.loc	1 78 9                          ; 00_naive_mfma.py:78:9
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[0:1]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[0:1]
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[38:41], a[38:39], a[82:83], v[38:41]
	.loc	1 78 9                          ; 00_naive_mfma.py:78:9
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[0:1]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[0:1]
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[38:41], a[42:43], a[86:87], v[38:41]
	.loc	1 78 9                          ; 00_naive_mfma.py:78:9
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[0:1]
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[0:1]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[0:1]
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[46:49], a[64:65], a[90:91], v[38:41]
	.loc	1 79 9                          ; 00_naive_mfma.py:79:9
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[0:1]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[0:1]
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[0:1]
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[38:41], a[10:11], a[78:79], v[50:53]
	.loc	1 79 9                          ; 00_naive_mfma.py:79:9
	v_lshl_add_u64 v[14:15], v[14:15], 0, s[0:1]
	v_lshl_add_u64 v[12:13], v[12:13], 0, s[0:1]
	v_lshl_add_u64 v[10:11], v[10:11], 0, s[0:1]
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[38:41], a[66:67], a[82:83], v[38:41]
	.loc	1 79 9                          ; 00_naive_mfma.py:79:9
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	v_lshl_add_u64 v[254:255], v[254:255], 0, s[0:1]
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_cmp_lg_u32 s15, s16
	.loc	1 76 15                         ; 00_naive_mfma.py:76:15
	v_mfma_f32_16x16x16_f16 v[38:41], a[70:71], a[86:87], v[38:41]
	v_mfma_f32_16x16x16_f16 v[50:53], a[74:75], a[90:91], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[12:13], a[78:79], v[54:57]
	v_mfma_f32_16x16x16_f16 v[38:41], a[68:69], a[82:83], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[72:73], a[86:87], v[38:41]
	v_mfma_f32_16x16x16_f16 v[54:57], a[76:77], a[90:91], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[4:5], a[78:79], v[58:61]
	v_mfma_f32_16x16x16_f16 v[38:41], a[94:95], a[82:83], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], a[86:87], v[38:41]
	v_mfma_f32_16x16x16_f16 v[58:61], a[102:103], a[90:91], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[6:7], a[78:79], v[62:65]
	v_mfma_f32_16x16x16_f16 v[38:41], a[96:97], a[82:83], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[100:101], a[86:87], v[38:41]
	v_mfma_f32_16x16x16_f16 v[62:65], a[104:105], a[90:91], v[38:41]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 v[38:41], a[0:1], a[78:79], v[66:69]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 v[38:41], a[106:107], a[82:83], v[38:41]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[38:41], a[114:115], a[86:87], v[38:41]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[66:69], a[110:111], a[90:91], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[2:3], a[78:79], v[70:73]
	v_mfma_f32_16x16x16_f16 v[38:41], a[108:109], a[82:83], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[116:117], a[86:87], v[38:41]
	v_mfma_f32_16x16x16_f16 v[70:73], a[112:113], a[90:91], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[32:33], a[80:81], v[74:77]
	v_mfma_f32_16x16x16_f16 v[38:41], a[36:37], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[40:41], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[74:77], a[62:63], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[34:35], a[80:81], v[78:81]
	v_mfma_f32_16x16x16_f16 v[38:41], a[38:39], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[42:43], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[78:81], a[64:65], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[10:11], a[80:81], v[82:85]
	v_mfma_f32_16x16x16_f16 v[38:41], a[66:67], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[70:71], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[82:85], a[74:75], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[12:13], a[80:81], v[86:89]
	v_mfma_f32_16x16x16_f16 v[38:41], a[68:69], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[72:73], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[86:89], a[76:77], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[4:5], a[80:81], v[90:93]
	v_mfma_f32_16x16x16_f16 v[38:41], a[94:95], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[90:93], a[102:103], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[6:7], a[80:81], v[94:97]
	v_mfma_f32_16x16x16_f16 v[38:41], a[96:97], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[100:101], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[94:97], a[104:105], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[0:1], a[80:81], v[98:101]
	v_mfma_f32_16x16x16_f16 v[38:41], a[106:107], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[114:115], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[98:101], a[110:111], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[2:3], a[80:81], v[102:105]
	v_mfma_f32_16x16x16_f16 v[38:41], a[108:109], a[84:85], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[116:117], a[88:89], v[38:41]
	v_mfma_f32_16x16x16_f16 v[102:105], a[112:113], a[92:93], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[32:33], a[46:47], v[106:109]
	v_mfma_f32_16x16x16_f16 v[38:41], a[36:37], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[40:41], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[106:109], a[62:63], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[34:35], a[46:47], v[110:113]
	v_mfma_f32_16x16x16_f16 v[38:41], a[38:39], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[42:43], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[110:113], a[64:65], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[10:11], a[46:47], v[114:117]
	v_mfma_f32_16x16x16_f16 v[38:41], a[66:67], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[70:71], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[114:117], a[74:75], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[12:13], a[46:47], v[118:121]
	v_mfma_f32_16x16x16_f16 v[38:41], a[68:69], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[72:73], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[118:121], a[76:77], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[4:5], a[46:47], v[122:125]
	v_mfma_f32_16x16x16_f16 v[38:41], a[94:95], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[122:125], a[102:103], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[6:7], a[46:47], v[126:129]
	v_mfma_f32_16x16x16_f16 v[38:41], a[96:97], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[100:101], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[126:129], a[104:105], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[0:1], a[46:47], v[130:133]
	v_mfma_f32_16x16x16_f16 v[38:41], a[106:107], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[114:115], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[130:133], a[110:111], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[2:3], a[46:47], v[134:137]
	v_mfma_f32_16x16x16_f16 v[38:41], a[108:109], a[50:51], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[116:117], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[134:137], a[112:113], a[58:59], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[32:33], a[48:49], v[138:141]
	v_mfma_f32_16x16x16_f16 v[38:41], a[36:37], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[40:41], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[138:141], a[62:63], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[34:35], a[48:49], v[142:145]
	v_mfma_f32_16x16x16_f16 v[38:41], a[38:39], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[42:43], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[142:145], a[64:65], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[10:11], a[48:49], v[146:149]
	v_mfma_f32_16x16x16_f16 v[38:41], a[66:67], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[70:71], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[146:149], a[74:75], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[12:13], a[48:49], v[150:153]
	v_mfma_f32_16x16x16_f16 v[38:41], a[68:69], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[72:73], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[150:153], a[76:77], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[4:5], a[48:49], v[154:157]
	v_mfma_f32_16x16x16_f16 v[38:41], a[94:95], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[154:157], a[102:103], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[6:7], a[48:49], v[158:161]
	v_mfma_f32_16x16x16_f16 v[38:41], a[96:97], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[100:101], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[158:161], a[104:105], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[0:1], a[48:49], v[162:165]
	v_mfma_f32_16x16x16_f16 v[38:41], a[106:107], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[114:115], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[162:165], a[110:111], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[2:3], a[48:49], v[166:169]
	v_mfma_f32_16x16x16_f16 v[38:41], a[108:109], a[52:53], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[116:117], a[56:57], v[38:41]
	v_mfma_f32_16x16x16_f16 v[166:169], a[112:113], a[60:61], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[32:33], a[28:29], v[170:173]
	v_mfma_f32_16x16x16_f16 v[38:41], a[36:37], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[40:41], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[170:173], a[62:63], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[34:35], a[28:29], v[174:177]
	v_mfma_f32_16x16x16_f16 v[38:41], a[38:39], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[42:43], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[174:177], a[64:65], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[10:11], a[28:29], v[178:181]
	v_mfma_f32_16x16x16_f16 v[38:41], a[66:67], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[70:71], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[178:181], a[74:75], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[12:13], a[28:29], v[182:185]
	v_mfma_f32_16x16x16_f16 v[38:41], a[68:69], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[72:73], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[182:185], a[76:77], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[4:5], a[28:29], v[186:189]
	v_mfma_f32_16x16x16_f16 v[38:41], a[94:95], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[186:189], a[102:103], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[6:7], a[28:29], v[190:193]
	v_mfma_f32_16x16x16_f16 v[38:41], a[96:97], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[100:101], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[190:193], a[104:105], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[0:1], a[28:29], v[194:197]
	v_mfma_f32_16x16x16_f16 v[38:41], a[106:107], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[114:115], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[194:197], a[110:111], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[2:3], a[28:29], v[198:201]
	v_mfma_f32_16x16x16_f16 v[38:41], a[108:109], a[24:25], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[116:117], a[20:21], v[38:41]
	v_mfma_f32_16x16x16_f16 v[198:201], a[112:113], a[16:17], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[32:33], a[30:31], v[202:205]
	v_mfma_f32_16x16x16_f16 v[38:41], a[36:37], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[40:41], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[202:205], a[62:63], a[18:19], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[34:35], a[30:31], v[206:209]
	v_mfma_f32_16x16x16_f16 v[38:41], a[38:39], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[42:43], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[206:209], a[64:65], a[18:19], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[10:11], a[30:31], v[210:213]
	v_mfma_f32_16x16x16_f16 v[38:41], a[66:67], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[70:71], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[210:213], a[74:75], a[18:19], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[12:13], a[30:31], v[214:217]
	v_mfma_f32_16x16x16_f16 v[38:41], a[68:69], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[72:73], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[214:217], a[76:77], a[18:19], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[4:5], a[30:31], v[218:221]
	v_mfma_f32_16x16x16_f16 v[38:41], a[94:95], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[218:221], a[102:103], a[18:19], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[6:7], a[30:31], v[222:225]
	v_mfma_f32_16x16x16_f16 v[38:41], a[96:97], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[100:101], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[2:5], a[32:33], a[118:119], v[2:5]
	v_mfma_f32_16x16x16_f16 v[242:245], a[34:35], a[118:119], v[242:245]
	v_mfma_f32_16x16x16_f16 v[226:229], a[10:11], a[118:119], v[226:229]
	v_mfma_f32_16x16x16_f16 v[246:249], a[12:13], a[118:119], v[246:249]
	v_mfma_f32_16x16x16_f16 v[230:233], a[4:5], a[118:119], v[230:233]
	v_mfma_f32_16x16x16_f16 v[250:253], a[6:7], a[118:119], v[250:253]
	v_mfma_f32_16x16x16_f16 v[234:237], a[0:1], a[118:119], v[234:237]
	v_mfma_f32_16x16x16_f16 a[148:151], a[2:3], a[118:119], a[148:151]
	v_mfma_f32_16x16x16_f16 a[152:155], a[32:33], a[120:121], a[152:155]
	v_mfma_f32_16x16x16_f16 a[156:159], a[34:35], a[120:121], a[156:159]
	v_mfma_f32_16x16x16_f16 a[160:163], a[10:11], a[120:121], a[160:163]
	v_mfma_f32_16x16x16_f16 a[164:167], a[12:13], a[120:121], a[164:167]
	v_mfma_f32_16x16x16_f16 a[168:171], a[4:5], a[120:121], a[168:171]
	v_mfma_f32_16x16x16_f16 a[172:175], a[6:7], a[120:121], a[172:175]
	v_mfma_f32_16x16x16_f16 a[176:179], a[0:1], a[120:121], a[176:179]
	v_mfma_f32_16x16x16_f16 a[180:183], a[2:3], a[120:121], a[180:183]
	v_mfma_f32_16x16x16_f16 a[184:187], a[32:33], a[78:79], a[184:187]
	v_mfma_f32_16x16x16_f16 v[222:225], a[104:105], a[18:19], v[38:41]
	v_mfma_f32_16x16x16_f16 v[6:9], a[0:1], a[30:31], v[6:9]
	v_mfma_f32_16x16x16_f16 v[38:41], a[2:3], a[30:31], v[238:241]
	v_mfma_f32_16x16x16_f16 v[2:5], a[36:37], a[122:123], v[2:5]
	v_mfma_f32_16x16x16_f16 v[242:245], a[38:39], a[122:123], v[242:245]
	v_mfma_f32_16x16x16_f16 v[226:229], a[66:67], a[122:123], v[226:229]
	v_mfma_f32_16x16x16_f16 v[246:249], a[68:69], a[122:123], v[246:249]
	v_mfma_f32_16x16x16_f16 v[230:233], a[94:95], a[122:123], v[230:233]
	v_mfma_f32_16x16x16_f16 v[250:253], a[96:97], a[122:123], v[250:253]
	v_mfma_f32_16x16x16_f16 v[234:237], a[106:107], a[122:123], v[234:237]
	v_mfma_f32_16x16x16_f16 a[148:151], a[108:109], a[122:123], a[148:151]
	v_mfma_f32_16x16x16_f16 a[152:155], a[36:37], a[124:125], a[152:155]
	v_mfma_f32_16x16x16_f16 a[156:159], a[38:39], a[124:125], a[156:159]
	v_mfma_f32_16x16x16_f16 a[160:163], a[66:67], a[124:125], a[160:163]
	v_mfma_f32_16x16x16_f16 a[164:167], a[68:69], a[124:125], a[164:167]
	v_mfma_f32_16x16x16_f16 a[168:171], a[94:95], a[124:125], a[168:171]
	v_mfma_f32_16x16x16_f16 a[172:175], a[96:97], a[124:125], a[172:175]
	v_mfma_f32_16x16x16_f16 a[176:179], a[106:107], a[124:125], a[176:179]
	v_mfma_f32_16x16x16_f16 a[180:183], a[108:109], a[124:125], a[180:183]
	v_mfma_f32_16x16x16_f16 a[184:187], a[36:37], a[82:83], a[184:187]
	v_mfma_f32_16x16x16_f16 v[6:9], a[106:107], a[26:27], v[6:9]
	v_mfma_f32_16x16x16_f16 v[38:41], a[108:109], a[26:27], v[38:41]
	v_mfma_f32_16x16x16_f16 v[2:5], a[40:41], a[126:127], v[2:5]
	v_mfma_f32_16x16x16_f16 v[242:245], a[42:43], a[126:127], v[242:245]
	v_mfma_f32_16x16x16_f16 v[226:229], a[70:71], a[126:127], v[226:229]
	v_mfma_f32_16x16x16_f16 v[246:249], a[72:73], a[126:127], v[246:249]
	v_mfma_f32_16x16x16_f16 v[230:233], a[98:99], a[126:127], v[230:233]
	v_mfma_f32_16x16x16_f16 v[250:253], a[100:101], a[126:127], v[250:253]
	v_mfma_f32_16x16x16_f16 v[234:237], a[114:115], a[126:127], v[234:237]
	v_mfma_f32_16x16x16_f16 a[148:151], a[116:117], a[126:127], a[148:151]
	v_mfma_f32_16x16x16_f16 a[152:155], a[40:41], a[128:129], a[152:155]
	v_mfma_f32_16x16x16_f16 a[156:159], a[42:43], a[128:129], a[156:159]
	v_mfma_f32_16x16x16_f16 a[160:163], a[70:71], a[128:129], a[160:163]
	v_mfma_f32_16x16x16_f16 a[164:167], a[72:73], a[128:129], a[164:167]
	v_mfma_f32_16x16x16_f16 a[168:171], a[98:99], a[128:129], a[168:171]
	v_mfma_f32_16x16x16_f16 a[172:175], a[100:101], a[128:129], a[172:175]
	v_mfma_f32_16x16x16_f16 a[176:179], a[114:115], a[128:129], a[176:179]
	v_mfma_f32_16x16x16_f16 a[180:183], a[116:117], a[128:129], a[180:183]
	v_mfma_f32_16x16x16_f16 a[184:187], a[40:41], a[86:87], a[184:187]
	v_mfma_f32_16x16x16_f16 v[6:9], a[114:115], a[22:23], v[6:9]
	v_mfma_f32_16x16x16_f16 v[38:41], a[116:117], a[22:23], v[38:41]
	v_mfma_f32_16x16x16_f16 v[2:5], a[62:63], a[130:131], v[2:5]
	v_mfma_f32_16x16x16_f16 v[242:245], a[64:65], a[130:131], v[242:245]
	v_mfma_f32_16x16x16_f16 v[226:229], a[74:75], a[130:131], v[226:229]
	v_mfma_f32_16x16x16_f16 v[246:249], a[76:77], a[130:131], v[246:249]
	v_mfma_f32_16x16x16_f16 v[230:233], a[102:103], a[130:131], v[230:233]
	v_mfma_f32_16x16x16_f16 v[250:253], a[104:105], a[130:131], v[250:253]
	v_mfma_f32_16x16x16_f16 v[234:237], a[110:111], a[130:131], v[234:237]
	v_mfma_f32_16x16x16_f16 a[148:151], a[112:113], a[130:131], a[148:151]
	v_mfma_f32_16x16x16_f16 a[152:155], a[62:63], a[132:133], a[152:155]
	v_mfma_f32_16x16x16_f16 a[156:159], a[64:65], a[132:133], a[156:159]
	v_mfma_f32_16x16x16_f16 a[160:163], a[74:75], a[132:133], a[160:163]
	v_mfma_f32_16x16x16_f16 a[164:167], a[76:77], a[132:133], a[164:167]
	v_mfma_f32_16x16x16_f16 a[168:171], a[102:103], a[132:133], a[168:171]
	v_mfma_f32_16x16x16_f16 a[172:175], a[104:105], a[132:133], a[172:175]
	v_mfma_f32_16x16x16_f16 a[176:179], a[110:111], a[132:133], a[176:179]
	v_mfma_f32_16x16x16_f16 a[180:183], a[112:113], a[132:133], a[180:183]
	v_mfma_f32_16x16x16_f16 a[184:187], a[62:63], a[90:91], a[184:187]
	v_mfma_f32_16x16x16_f16 v[6:9], a[110:111], a[18:19], v[6:9]
	v_mfma_f32_16x16x16_f16 v[238:241], a[112:113], a[18:19], v[38:41]
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_cbranch_scc0 .LBB0_36
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 71 58                         ; 00_naive_mfma.py:71:58
	s_lshl_b32 s2, s15, 6
	.loc	1 71 54 is_stmt 0               ; 00_naive_mfma.py:71:54
	s_sub_i32 s2, s10, s2
	v_accvgpr_read_b32 v38, a138
	.loc	1 71 35                         ; 00_naive_mfma.py:71:35
	v_cmp_gt_i32_e32 vcc, s2, v38
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[18:21], v[36:37], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a48, a20
	v_accvgpr_mov_b32 a49, a21
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[20:23], v[34:35], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a50, a22
	v_accvgpr_mov_b32 a51, a23
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_9
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[24:27], v[32:33], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a52, a26
	v_accvgpr_mov_b32 a53, a27
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[26:29], v[30:31], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a54, a28
	v_accvgpr_mov_b32 a55, a29
.LBB0_11:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[30:33], v[28:29], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a56, a32
	v_accvgpr_mov_b32 a57, a33
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_15
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[32:35], v[26:27], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a58, a34
	v_accvgpr_mov_b32 a59, a35
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[36:39], v[24:25], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a62, a38
	v_accvgpr_mov_b32 a63, a39
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	.loc	1 71 14                         ; 00_naive_mfma.py:71:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[38:41], v[22:23], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a64, a40
	v_accvgpr_mov_b32 a65, a41
.LBB0_19:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	.loc	1 72 14 is_stmt 1               ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[42:45], v[20:21], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a70, a44
	v_accvgpr_mov_b32 a71, a45
.LBB0_21:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14 is_stmt 0                ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[44:47], v[18:19], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a72, a46
	v_accvgpr_mov_b32 a73, a47
.LBB0_23:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[12:15], v[16:17], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a66, a14
	v_accvgpr_mov_b32 a67, a15
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[14:17], v[14:15], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a68, a16
	v_accvgpr_mov_b32 a69, a17
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[6:9], v[12:13], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a94, a8
	v_accvgpr_mov_b32 a95, a9
.LBB0_29:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[8:11], v[10:11], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a96, a10
	v_accvgpr_mov_b32 a97, a11
.LBB0_31:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[0:3], v[0:1], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a106, a2
	v_accvgpr_mov_b32 a107, a3
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	.loc	1 72 14                         ; 00_naive_mfma.py:72:14
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	global_load_dwordx4 a[2:5], v[254:255], off
	s_waitcnt vmcnt(0)
	v_accvgpr_mov_b32 a108, a4
	v_accvgpr_mov_b32 a109, a5
	s_branch .LBB0_2
.LBB0_35:
	.loc	1 0 14                          ; 00_naive_mfma.py:0:14
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v129, 0
	s_branch .LBB0_37
.LBB0_36:                               ; %._crit_edge.loopexit
	.loc	1 81 9 is_stmt 1                ; 00_naive_mfma.py:81:9
	v_cvt_f16_f32_e32 v0, v242
	v_accvgpr_read_b32 v36, a148
	v_accvgpr_read_b32 v37, a149
	v_cvt_f16_f32_e32 v10, v37
	v_accvgpr_write_b32 a0, v0
	v_cvt_f16_f32_e32 v0, v243
	v_accvgpr_read_b32 v40, a152
	v_accvgpr_write_b32 a18, v10
	v_cvt_f16_f32_e32 v10, v40
	v_accvgpr_write_b32 a2, v0
	v_cvt_f16_f32_e32 v0, v226
	v_accvgpr_read_b32 v41, a153
	v_accvgpr_write_b32 a3, v10
	v_cvt_f16_f32_e32 v10, v41
	v_accvgpr_write_b32 a4, v0
	v_cvt_f16_f32_e32 v0, v227
	v_accvgpr_read_b32 v42, a154
	v_accvgpr_read_b32 v43, a155
	v_cvt_f16_f32_e32 v41, v42
	v_accvgpr_write_b32 a8, v0
	v_cvt_f16_f32_e32 v0, v246
	v_cvt_f16_f32_e32 v40, v43
	v_accvgpr_read_b32 v42, a156
	v_accvgpr_write_b32 a19, v10
	v_accvgpr_write_b32 a6, v0
	v_cvt_f16_f32_e32 v0, v230
	v_cvt_f16_f32_e32 v10, v42
	v_accvgpr_read_b32 v43, a157
	v_accvgpr_read_b32 v44, a158
	v_accvgpr_write_b32 a9, v0
	v_cvt_f16_f32_e32 v0, v231
	v_accvgpr_read_b32 v45, a159
	v_cvt_f16_f32_e32 v23, v249
	v_cvt_f16_f32_e32 v11, v43
	v_accvgpr_write_b32 a13, v0
	v_cvt_f16_f32_e32 v0, v250
	v_cvt_f16_f32_e32 v249, v44
	v_cvt_f16_f32_e32 v250, v45
	v_accvgpr_read_b32 v42, a160
	v_accvgpr_write_b32 a5, v10
	v_cvt_f16_f32_e32 v10, v42
	v_accvgpr_read_b32 v43, a161
	v_accvgpr_read_b32 v44, a162
	v_accvgpr_read_b32 v45, a163
	v_accvgpr_write_b32 a11, v0
	v_cvt_f16_f32_e32 v0, v251
	v_cvt_f16_f32_e32 v29, v252
	v_accvgpr_write_b32 a7, v10
	v_cvt_f16_f32_e32 v10, v43
	v_cvt_f16_f32_e32 v251, v44
	v_cvt_f16_f32_e32 v252, v45
	v_accvgpr_read_b32 v42, a164
	v_cvt_f16_f32_e32 v12, v42
	v_accvgpr_read_b32 v38, a150
	v_accvgpr_read_b32 v43, a165
	v_accvgpr_read_b32 v44, a166
	v_accvgpr_read_b32 v45, a167
	v_cvt_f16_f32_e32 v31, v253
	v_cvt_f16_f32_e32 v37, v38
	v_accvgpr_write_b32 a10, v12
	v_cvt_f16_f32_e32 v12, v43
	v_cvt_f16_f32_e32 v253, v44
	v_cvt_f16_f32_e32 v38, v45
	v_accvgpr_read_b32 v42, a168
	v_cvt_f16_f32_e32 v14, v42
	v_accvgpr_read_b32 v43, a169
	v_accvgpr_read_b32 v44, a170
	v_accvgpr_read_b32 v45, a171
	v_accvgpr_write_b32 a12, v14
	v_cvt_f16_f32_e32 v255, v43
	v_cvt_f16_f32_e32 v14, v44
	v_cvt_f16_f32_e32 v16, v45
	v_accvgpr_read_b32 v42, a172
	v_cvt_f16_f32_e32 v18, v42
	v_accvgpr_read_b32 v43, a173
	v_accvgpr_read_b32 v44, a174
	v_accvgpr_read_b32 v45, a175
	v_accvgpr_write_b32 a15, v18
	v_cvt_f16_f32_e32 v254, v43
	v_cvt_f16_f32_e32 v18, v44
	v_cvt_f16_f32_e32 v20, v45
	v_accvgpr_read_b32 v42, a176
	v_cvt_f16_f32_e32 v22, v42
	v_accvgpr_write_b32 a1, v0
	v_cvt_f16_f32_e32 v0, v234
	v_accvgpr_read_b32 v43, a177
	v_accvgpr_read_b32 v44, a178
	v_accvgpr_read_b32 v45, a179
	v_cvt_f16_f32_e32 v21, v248
	v_accvgpr_write_b32 a17, v22
	v_cvt_f16_f32_e32 v248, v43
	v_cvt_f16_f32_e32 v22, v44
	v_cvt_f16_f32_e32 v24, v45
	v_accvgpr_read_b32 v42, a180
	v_cvt_f16_f32_e32 v26, v42
	v_accvgpr_write_b32 a14, v0
	v_cvt_f16_f32_e32 v0, v235
	v_accvgpr_read_b32 v43, a181
	v_accvgpr_read_b32 v44, a182
	v_accvgpr_read_b32 v45, a183
	v_cvt_f16_f32_e32 v1, v247
	v_accvgpr_write_b32 a20, v26
	v_cvt_f16_f32_e32 v247, v43
	v_cvt_f16_f32_e32 v26, v44
	v_cvt_f16_f32_e32 v28, v45
	v_accvgpr_read_b32 v42, a184
	v_accvgpr_read_b32 v39, a151
	v_accvgpr_read_b32 v43, a185
	v_accvgpr_read_b32 v44, a186
	v_accvgpr_read_b32 v45, a187
	v_cvt_f16_f32_e32 v13, v244
	v_cvt_f16_f32_e32 v15, v245
	v_cvt_f16_f32_e32 v17, v228
	v_cvt_f16_f32_e32 v19, v229
	v_cvt_f16_f32_e32 v25, v232
	v_cvt_f16_f32_e32 v27, v233
	v_accvgpr_write_b32 a16, v0
	v_cvt_f16_f32_e32 v33, v236
	v_cvt_f16_f32_e32 v35, v237
	v_cvt_f16_f32_e32 v0, v36
	v_cvt_f16_f32_e32 v39, v39
	v_cvt_f16_f32_e32 v246, v42
	v_cvt_f16_f32_e32 v245, v43
	v_cvt_f16_f32_e32 v43, v44
	v_cvt_f16_f32_e32 v45, v45
	v_cvt_f16_f32_e32 v244, v46
	v_cvt_f16_f32_e32 v243, v47
	v_cvt_f16_f32_e32 v47, v48
	v_cvt_f16_f32_e32 v49, v49
	v_cvt_f16_f32_e32 v242, v50
	v_cvt_f16_f32_e32 v237, v51
	v_cvt_f16_f32_e32 v51, v52
	v_cvt_f16_f32_e32 v53, v53
	v_cvt_f16_f32_e32 v236, v54
	v_cvt_f16_f32_e32 v235, v55
	v_cvt_f16_f32_e32 v55, v56
	v_cvt_f16_f32_e32 v57, v57
	v_cvt_f16_f32_e32 v234, v58
	v_cvt_f16_f32_e32 v233, v59
	v_cvt_f16_f32_e32 v59, v60
	v_cvt_f16_f32_e32 v61, v61
	v_cvt_f16_f32_e32 v232, v62
	v_cvt_f16_f32_e32 v231, v63
	v_cvt_f16_f32_e32 v63, v64
	v_cvt_f16_f32_e32 v65, v65
	v_cvt_f16_f32_e32 v230, v66
	v_cvt_f16_f32_e32 v229, v67
	v_cvt_f16_f32_e32 v67, v68
	v_cvt_f16_f32_e32 v69, v69
	v_cvt_f16_f32_e32 v228, v70
	v_cvt_f16_f32_e32 v227, v71
	v_cvt_f16_f32_e32 v71, v72
	v_cvt_f16_f32_e32 v73, v73
	v_cvt_f16_f32_e32 v226, v74
	v_cvt_f16_f32_e32 v36, v75
	v_cvt_f16_f32_e32 v75, v76
	v_cvt_f16_f32_e32 v77, v77
	v_cvt_f16_f32_e32 v34, v78
	v_cvt_f16_f32_e32 v32, v79
	v_cvt_f16_f32_e32 v78, v80
	v_cvt_f16_f32_e32 v80, v81
	v_cvt_f16_f32_e32 v30, v82
	v_cvt_f16_f32_e32 v42, v83
	v_cvt_f16_f32_e32 v81, v84
	v_cvt_f16_f32_e32 v82, v85
	v_cvt_f16_f32_e32 v46, v86
	v_cvt_f16_f32_e32 v44, v87
	v_cvt_f16_f32_e32 v84, v88
	v_cvt_f16_f32_e32 v86, v89
	v_cvt_f16_f32_e32 v50, v90
	v_cvt_f16_f32_e32 v48, v91
	v_cvt_f16_f32_e32 v88, v92
	v_cvt_f16_f32_e32 v90, v93
	v_cvt_f16_f32_e32 v54, v94
	v_cvt_f16_f32_e32 v52, v95
	v_cvt_f16_f32_e32 v92, v96
	v_cvt_f16_f32_e32 v94, v97
	v_cvt_f16_f32_e32 v58, v98
	v_cvt_f16_f32_e32 v56, v99
	v_cvt_f16_f32_e32 v96, v100
	v_cvt_f16_f32_e32 v98, v101
	v_cvt_f16_f32_e32 v62, v102
	v_cvt_f16_f32_e32 v60, v103
	v_cvt_f16_f32_e32 v100, v104
	v_cvt_f16_f32_e32 v102, v105
	v_cvt_f16_f32_e32 v66, v106
	v_cvt_f16_f32_e32 v64, v107
	v_cvt_f16_f32_e32 v104, v108
	v_cvt_f16_f32_e32 v106, v109
	v_cvt_f16_f32_e32 v70, v110
	v_cvt_f16_f32_e32 v68, v111
	v_cvt_f16_f32_e32 v108, v112
	v_cvt_f16_f32_e32 v110, v113
	v_cvt_f16_f32_e32 v74, v114
	v_cvt_f16_f32_e32 v72, v115
	v_cvt_f16_f32_e32 v112, v116
	v_cvt_f16_f32_e32 v114, v117
	v_cvt_f16_f32_e32 v76, v118
	v_cvt_f16_f32_e32 v93, v119
	v_cvt_f16_f32_e32 v116, v120
	v_cvt_f16_f32_e32 v118, v121
	v_cvt_f16_f32_e32 v91, v122
	v_cvt_f16_f32_e32 v89, v123
	v_cvt_f16_f32_e32 v120, v124
	v_cvt_f16_f32_e32 v122, v125
	v_cvt_f16_f32_e32 v87, v126
	v_cvt_f16_f32_e32 v85, v127
	v_cvt_f16_f32_e32 v124, v128
	v_cvt_f16_f32_e32 v126, v129
	v_cvt_f16_f32_e32 v128, v132
	v_cvt_f16_f32_e32 v132, v133
	v_cvt_f16_f32_e32 v83, v135
	v_cvt_f16_f32_e32 v133, v136
	v_cvt_f16_f32_e32 v136, v137
	v_cvt_f16_f32_e32 v137, v139
	v_cvt_f16_f32_e32 v140, v140
	v_cvt_f16_f32_e32 v141, v141
	v_cvt_f16_f32_e32 v139, v143
	v_cvt_f16_f32_e32 v144, v144
	v_cvt_f16_f32_e32 v143, v145
	v_cvt_f16_f32_e32 v145, v147
	v_cvt_f16_f32_e32 v147, v148
	v_cvt_f16_f32_e32 v148, v149
	v_cvt_f16_f32_e32 v149, v150
	v_cvt_f16_f32_e32 v152, v152
	v_cvt_f16_f32_e32 v150, v153
	v_cvt_f16_f32_e32 v153, v154
	v_cvt_f16_f32_e32 v154, v156
	v_cvt_f16_f32_e32 v157, v157
	v_cvt_f16_f32_e32 v156, v158
	v_cvt_f16_f32_e32 v158, v160
	v_cvt_f16_f32_e32 v160, v161
	v_cvt_f16_f32_e32 v161, v162
	v_cvt_f16_f32_e32 v164, v164
	v_cvt_f16_f32_e32 v165, v165
	v_cvt_f16_f32_e32 v162, v166
	v_cvt_f16_f32_e32 v166, v167
	v_cvt_f16_f32_e32 v168, v168
	v_cvt_f16_f32_e32 v169, v169
	v_cvt_f16_f32_e32 v167, v170
	v_cvt_f16_f32_e32 v172, v172
	v_cvt_f16_f32_e32 v173, v173
	v_cvt_f16_f32_e32 v170, v174
	v_cvt_f16_f32_e32 v176, v176
	v_cvt_f16_f32_e32 v177, v177
	v_cvt_f16_f32_e32 v174, v179
	v_cvt_f16_f32_e32 v180, v180
	v_cvt_f16_f32_e32 v181, v181
	v_cvt_f16_f32_e32 v179, v183
	v_cvt_f16_f32_e32 v183, v184
	v_cvt_f16_f32_e32 v185, v185
	v_cvt_f16_f32_e32 v184, v186
	v_cvt_f16_f32_e32 v79, v187
	v_cvt_f16_f32_e32 v186, v188
	v_cvt_f16_f32_e32 v187, v189
	v_cvt_f16_f32_e32 v189, v190
	v_cvt_f16_f32_e32 v188, v191
	v_cvt_f16_f32_e32 v192, v192
	v_cvt_f16_f32_e32 v191, v193
	v_cvt_f16_f32_e32 v190, v194
	v_cvt_f16_f32_e32 v193, v196
	v_cvt_f16_f32_e32 v197, v197
	v_cvt_f16_f32_e32 v196, v198
	v_cvt_f16_f32_e32 v194, v199
	v_cvt_f16_f32_e32 v200, v200
	v_cvt_f16_f32_e32 v201, v201
	v_cvt_f16_f32_e32 v199, v202
	v_cvt_f16_f32_e32 v198, v203
	v_cvt_f16_f32_e32 v202, v204
	v_cvt_f16_f32_e32 v203, v205
	v_cvt_f16_f32_e32 v205, v206
	v_cvt_f16_f32_e32 v206, v207
	v_cvt_f16_f32_e32 v208, v208
	v_cvt_f16_f32_e32 v209, v209
	v_cvt_f16_f32_e32 v204, v211
	v_cvt_f16_f32_e32 v211, v212
	v_cvt_f16_f32_e32 v212, v213
	v_cvt_f16_f32_e32 v207, v214
	v_cvt_f16_f32_e32 v216, v216
	v_cvt_f16_f32_e32 v217, v217
	v_cvt_f16_f32_e32 v214, v218
	v_cvt_f16_f32_e32 v213, v219
	v_cvt_f16_f32_e32 v218, v220
	v_cvt_f16_f32_e32 v219, v221
	v_cvt_f16_f32_e32 v221, v222
	v_cvt_f16_f32_e32 v220, v223
	v_cvt_f16_f32_e32 v222, v224
	v_cvt_f16_f32_e32 v223, v225
	v_cvt_f16_f32_e32 v7, v7
	v_cvt_f16_f32_e32 v224, v8
	v_cvt_f16_f32_e32 v225, v9
	v_cvt_f16_f32_e32 v240, v240
	v_cvt_f16_f32_e32 v241, v241
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_cvt_f16_f32_e32 v130, v130
	v_cvt_f16_f32_e32 v131, v131
	v_cvt_f16_f32_e32 v134, v134
	v_cvt_f16_f32_e32 v138, v138
	v_cvt_f16_f32_e32 v142, v142
	v_cvt_f16_f32_e32 v146, v146
	v_cvt_f16_f32_e32 v151, v151
	v_cvt_f16_f32_e32 v155, v155
	v_cvt_f16_f32_e32 v159, v159
	v_cvt_f16_f32_e32 v163, v163
	v_cvt_f16_f32_e32 v171, v171
	v_cvt_f16_f32_e32 v175, v175
	v_cvt_f16_f32_e32 v178, v178
	v_cvt_f16_f32_e32 v182, v182
	v_cvt_f16_f32_e32 v195, v195
	v_cvt_f16_f32_e32 v210, v210
	v_cvt_f16_f32_e32 v215, v215
	v_cvt_f16_f32_e32 v135, v6
	v_cvt_f16_f32_e32 v238, v238
	v_cvt_f16_f32_e32 v239, v239
	v_cvt_f16_f32_e32 v8, v2
	v_cvt_f16_f32_e32 v6, v3
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v129, v4, v5
	v_pack_b32_f16 v127, v13, v15
	v_pack_b32_f16 v125, v17, v19
	v_pack_b32_f16 v123, v21, v23
	v_pack_b32_f16 v121, v25, v27
	v_pack_b32_f16 v119, v29, v31
	v_pack_b32_f16 v117, v33, v35
	v_pack_b32_f16 v115, v37, v39
	v_pack_b32_f16 v113, v41, v40
	v_mov_b32_e32 v40, v93
	v_pack_b32_f16 v111, v249, v250
	v_pack_b32_f16 v109, v251, v252
	v_pack_b32_f16 v107, v253, v38
	v_pack_b32_f16 v105, v14, v16
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v16, v79
	v_pack_b32_f16 v103, v18, v20
	v_mov_b32_e32 v20, v85
	v_mov_b32_e32 v18, v83
	v_pack_b32_f16 v101, v22, v24
	v_mov_b32_e32 v24, v87
	v_pack_b32_f16 v99, v26, v28
	v_mov_b32_e32 v28, v91
	v_mov_b32_e32 v26, v89
	v_pack_b32_f16 v97, v43, v45
	v_pack_b32_f16 v95, v47, v49
	v_pack_b32_f16 v93, v51, v53
	v_pack_b32_f16 v91, v55, v57
	v_pack_b32_f16 v89, v59, v61
	v_pack_b32_f16 v87, v63, v65
	v_pack_b32_f16 v85, v67, v69
	v_pack_b32_f16 v83, v71, v73
	v_pack_b32_f16 v79, v75, v77
	v_pack_b32_f16 v77, v78, v80
	v_pack_b32_f16 v75, v81, v82
	v_pack_b32_f16 v73, v84, v86
	v_pack_b32_f16 v71, v88, v90
	v_pack_b32_f16 v69, v92, v94
	v_pack_b32_f16 v67, v96, v98
	v_pack_b32_f16 v65, v100, v102
	v_pack_b32_f16 v63, v104, v106
	v_mov_b32_e32 v106, v12
	v_accvgpr_read_b32 v12, a18
	v_pack_b32_f16 v61, v108, v110
	v_mov_b32_e32 v110, v11
	v_mov_b32_e32 v108, v10
	v_accvgpr_read_b32 v10, a8
	v_accvgpr_read_b32 v1, a13
	v_pack_b32_f16 v59, v112, v114
	v_accvgpr_read_b32 v114, a16
	v_accvgpr_read_b32 v112, a19
	v_pack_b32_f16 v57, v116, v118
	v_accvgpr_read_b32 v118, a11
	v_accvgpr_read_b32 v116, a14
	v_pack_b32_f16 v55, v120, v122
	v_accvgpr_read_b32 v122, a6
	v_accvgpr_read_b32 v120, a9
	v_pack_b32_f16 v53, v124, v126
	v_accvgpr_read_b32 v126, a2
	v_accvgpr_read_b32 v124, a4
	v_pack_b32_f16 v51, v128, v132
	v_mov_b32_e32 v132, v7
	v_pack_b32_f16 v49, v133, v136
	v_mov_b32_e32 v136, v72
	v_mov_b32_e32 v133, v76
	v_pack_b32_f16 v47, v140, v141
	v_pack_b32_f16 v45, v144, v143
	v_mov_b32_e32 v144, v68
	v_mov_b32_e32 v143, v74
	v_pack_b32_f16 v43, v147, v148
	v_mov_b32_e32 v148, v64
	v_mov_b32_e32 v64, v60
	v_mov_b32_e32 v60, v62
	v_mov_b32_e32 v62, v56
	v_mov_b32_e32 v56, v58
	v_mov_b32_e32 v58, v52
	v_mov_b32_e32 v52, v54
	v_mov_b32_e32 v54, v48
	v_mov_b32_e32 v48, v50
	v_mov_b32_e32 v50, v44
	v_mov_b32_e32 v44, v46
	v_mov_b32_e32 v46, v42
	v_mov_b32_e32 v42, v30
	v_mov_b32_e32 v30, v32
	v_mov_b32_e32 v32, v34
	v_mov_b32_e32 v34, v36
	v_mov_b32_e32 v147, v70
	v_pack_b32_f16 v41, v152, v150
	v_mov_b32_e32 v150, v66
	v_pack_b32_f16 v39, v154, v157
	v_accvgpr_read_b32 v157, a20
	v_accvgpr_read_b32 v11, a1
	v_pack_b32_f16 v37, v158, v160
	v_accvgpr_read_b32 v160, a15
	v_accvgpr_read_b32 v158, a17
	v_pack_b32_f16 v35, v164, v165
	v_accvgpr_read_b32 v165, a10
	v_accvgpr_read_b32 v164, a12
	v_pack_b32_f16 v33, v168, v169
	v_accvgpr_read_b32 v169, a5
	v_accvgpr_read_b32 v168, a7
	v_pack_b32_f16 v31, v172, v173
	v_accvgpr_read_b32 v173, a0
	v_accvgpr_read_b32 v172, a3
	v_pack_b32_f16 v29, v176, v177
	v_pack_b32_f16 v27, v180, v181
	v_pack_b32_f16 v25, v183, v185
	v_pack_b32_f16 v23, v186, v187
	v_pack_b32_f16 v21, v192, v191
	v_pack_b32_f16 v19, v193, v197
	v_pack_b32_f16 v17, v200, v201
	v_pack_b32_f16 v15, v202, v203
	v_pack_b32_f16 v13, v208, v209
	v_pack_b32_f16 v185, v211, v212
	v_pack_b32_f16 v9, v216, v217
	v_pack_b32_f16 v7, v218, v219
	v_pack_b32_f16 v5, v222, v223
	v_pack_b32_f16 v3, v224, v225
	v_pack_b32_f16 v183, v240, v241
	v_accvgpr_read_b32 v78, a134
	v_accvgpr_read_b32 v140, a136
	v_accvgpr_read_b32 v22, a135
	v_accvgpr_read_b32 v38, a137
.LBB0_37:                               ; %._crit_edge
	.loc	1 85 15                         ; 00_naive_mfma.py:85:15
	v_and_b32_e32 v2, 15, v78
	s_lshr_b32 s0, s14, 3
	.loc	1 87 22                         ; 00_naive_mfma.py:87:22
	v_mul_lo_u32 v80, v22, s13
	.loc	1 85 15                         ; 00_naive_mfma.py:85:15
	v_and_or_b32 v4, s0, 16, v2
	.loc	1 86 15                         ; 00_naive_mfma.py:86:15
	v_lshrrev_b32_e32 v2, 2, v38
	.loc	1 87 14                         ; 00_naive_mfma.py:87:14
	v_ashrrev_i32_e32 v81, 31, v80
	.loc	1 86 15                         ; 00_naive_mfma.py:86:15
	v_and_b32_e32 v2, 28, v2
	.loc	1 87 14                         ; 00_naive_mfma.py:87:14
	v_lshlrev_b64 v[80:81], 1, v[80:81]
	v_ashrrev_i32_e32 v141, 31, v140
	v_lshl_add_u64 v[80:81], s[6:7], 0, v[80:81]
	v_lshlrev_b64 v[186:187], 1, v[140:141]
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v4
	.loc	1 90 40 is_stmt 0               ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e32 vcc, s9, v2
	.loc	1 87 14 is_stmt 1               ; 00_naive_mfma.py:87:14
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[186:187]
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v78, v4, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_39
; %bb.38:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v186, v78, v2
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_mov_b32 s2, 0x5040100
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v187, 31, v186
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	v_perm_b32 v128, v6, v8, s2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_lshl_add_u64 v[186:187], v[186:187], 1, v[80:81]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[186:187], v[128:129], off
.LBB0_39:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v6, 32, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[0:1], s9, v6
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[4:5], s[14:15], s[0:1]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_41
; %bb.40:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v128, v6, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v129, 31, v128
	s_mov_b32 s4, 0x5040100
	v_lshl_add_u64 v[128:129], v[128:129], 1, v[80:81]
	v_perm_b32 v126, v126, v173, s4
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[128:129], v[126:127], off
.LBB0_41:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[2:3]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v8, 64, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[2:3], s9, v8
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[6:7], s[14:15], s[2:3]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB0_43
; %bb.42:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v126, v8, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v127, 31, v126
	s_mov_b32 s6, 0x5040100
	v_lshl_add_u64 v[126:127], v[126:127], 1, v[80:81]
	v_perm_b32 v124, v10, v124, s6
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[126:127], v[124:125], off
.LBB0_43:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[4:5]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v10, 0x60, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[4:5], s9, v10
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[10:11], s[14:15], s[4:5]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[6:7], s[10:11]
	s_cbranch_execz .LBB0_45
; %bb.44:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v124, v10, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v125, 31, v124
	s_mov_b32 s10, 0x5040100
	v_lshl_add_u64 v[124:125], v[124:125], 1, v[80:81]
	v_perm_b32 v122, v0, v122, s10
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[124:125], v[122:123], off
.LBB0_45:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[6:7]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v0, 0x80, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[6:7], s9, v0
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[16:17], s[14:15], s[6:7]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[10:11], s[16:17]
	s_cbranch_execz .LBB0_47
; %bb.46:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v122, v0, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v123, 31, v122
	s_mov_b32 s12, 0x5040100
	v_lshl_add_u64 v[122:123], v[122:123], 1, v[80:81]
	v_perm_b32 v120, v1, v120, s12
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[122:123], v[120:121], off
.LBB0_47:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[10:11]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v1, 0xa0, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[18:19], s9, v1
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[16:17], s[14:15], s[18:19]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[10:11], s[16:17]
	s_cbranch_execz .LBB0_49
; %bb.48:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v120, v1, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v121, 31, v120
	s_mov_b32 s12, 0x5040100
	v_lshl_add_u64 v[120:121], v[120:121], 1, v[80:81]
	v_perm_b32 v118, v11, v118, s12
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[120:121], v[118:119], off
.LBB0_49:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[10:11]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v11, 0xc0, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[10:11], s9, v11
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[10:11]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[16:17], s[20:21]
	s_cbranch_execz .LBB0_51
; %bb.50:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v118, v11, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v119, 31, v118
	s_mov_b32 s12, 0x5040100
	v_lshl_add_u64 v[118:119], v[118:119], 1, v[80:81]
	v_perm_b32 v116, v114, v116, s12
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[118:119], v[116:117], off
.LBB0_51:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[16:17]
	.loc	1 86 15 is_stmt 1               ; 00_naive_mfma.py:86:15
	v_or_b32_e32 v116, 0xe0, v2
	.loc	1 90 40                         ; 00_naive_mfma.py:90:40
	v_cmp_gt_i32_e64 s[16:17], s9, v116
	.loc	1 90 14 is_stmt 0               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5 is_stmt 1                ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_53
; %bb.52:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v118, v116, v78
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v119, 31, v118
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[118:119], v[118:119], 1, v[80:81]
	v_perm_b32 v114, v12, v14, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[118:119], v[114:115], off
.LBB0_53:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v14, 32, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_116
; %bb.54:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_117
.LBB0_55:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_118
.LBB0_56:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_119
.LBB0_57:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_120
.LBB0_58:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_121
.LBB0_59:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_122
.LBB0_60:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_62
.LBB0_61:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v100, v12, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v101, 31, v100
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[100:101], v[100:101], 1, v[80:81]
	v_perm_b32 v98, v247, v157, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[100:101], v[98:99], off
.LBB0_62:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v14, 64, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_123
; %bb.63:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_124
.LBB0_64:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_125
.LBB0_65:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_126
.LBB0_66:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_127
.LBB0_67:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_128
.LBB0_68:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_129
.LBB0_69:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_71
.LBB0_70:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v84, v12, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v85, 31, v84
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[84:85], v[84:85], 1, v[80:81]
	v_perm_b32 v82, v227, v228, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[84:85], v[82:83], off
.LBB0_71:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v14, 0x60, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_130
; %bb.72:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_131
.LBB0_73:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_132
.LBB0_74:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_133
.LBB0_75:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_134
.LBB0_76:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_135
.LBB0_77:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_136
.LBB0_78:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_80
.LBB0_79:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v66, v12, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v67, 31, v66
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[80:81]
	v_perm_b32 v64, v64, v60, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[66:67], v[64:65], off
.LBB0_80:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v14, 0x80, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_137
; %bb.81:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_138
.LBB0_82:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_139
.LBB0_83:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_140
.LBB0_84:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_141
.LBB0_85:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_142
.LBB0_86:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_143
.LBB0_87:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_89
.LBB0_88:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v50, v12, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v51, 31, v50
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[80:81]
	v_perm_b32 v48, v18, v134, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[50:51], v[48:49], off
.LBB0_89:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v14, 0xa0, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_144
; %bb.90:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_145
.LBB0_91:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_146
.LBB0_92:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_147
.LBB0_93:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_148
.LBB0_94:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_149
.LBB0_95:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_150
.LBB0_96:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_98
.LBB0_97:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v34, v12, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v35, 31, v34
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[80:81]
	v_perm_b32 v32, v166, v162, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[34:35], v[32:33], off
.LBB0_98:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v14, 0xc0, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v14
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_151
; %bb.99:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_152
.LBB0_100:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_153
.LBB0_101:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_154
.LBB0_102:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_155
.LBB0_103:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_156
.LBB0_104:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execnz .LBB0_157
.LBB0_105:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execz .LBB0_107
.LBB0_106:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v18, v12, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v19, 31, v18
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[80:81]
	v_perm_b32 v16, v194, v196, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[18:19], v[16:17], off
.LBB0_107:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 85 15 is_stmt 1               ; 00_naive_mfma.py:85:15
	v_or_b32_e32 v4, 0xe0, v4
	.loc	1 90 15                         ; 00_naive_mfma.py:90:15
	v_cmp_gt_i32_e64 s[14:15], s8, v4
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_mul_lo_u32 v16, v4, s13
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[12:13], s[14:15], vcc
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[8:9], s[12:13]
	s_cbranch_execnz .LBB0_158
; %bb.108:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[8:9]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[8:9], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execnz .LBB0_159
.LBB0_109:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_160
.LBB0_110:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_161
.LBB0_111:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_162
.LBB0_112:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_163
.LBB0_113:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_164
.LBB0_114:
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14 is_stmt 1               ; 00_naive_mfma.py:90:14
	s_and_b64 s[0:1], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execnz .LBB0_165
.LBB0_115:
	.loc	1 8 1                           ; 00_naive_mfma.py:8:1
	s_endpgm
.LBB0_116:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v114, v12, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v115, 31, v114
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[80:81]
	v_perm_b32 v112, v112, v172, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[114:115], v[112:113], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_55
.LBB0_117:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v112, v12, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v113, 31, v112
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[112:113], v[112:113], 1, v[80:81]
	v_perm_b32 v110, v110, v169, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[112:113], v[110:111], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_56
.LBB0_118:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v110, v12, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v111, 31, v110
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[110:111], v[110:111], 1, v[80:81]
	v_perm_b32 v108, v108, v168, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[110:111], v[108:109], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_57
.LBB0_119:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v108, v12, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v109, 31, v108
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[108:109], v[108:109], 1, v[80:81]
	v_perm_b32 v106, v106, v165, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[108:109], v[106:107], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_58
.LBB0_120:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v106, v12, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v107, 31, v106
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[106:107], v[106:107], 1, v[80:81]
	v_perm_b32 v104, v255, v164, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[106:107], v[104:105], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_59
.LBB0_121:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v104, v12, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v105, 31, v104
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[104:105], v[104:105], 1, v[80:81]
	v_perm_b32 v102, v254, v160, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[104:105], v[102:103], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_60
.LBB0_122:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v102, v12, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v103, 31, v102
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[102:103], v[102:103], 1, v[80:81]
	v_perm_b32 v100, v248, v158, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[102:103], v[100:101], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_61
	s_branch .LBB0_62
.LBB0_123:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v98, v12, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v99, 31, v98
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[80:81]
	v_perm_b32 v96, v245, v246, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[98:99], v[96:97], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_64
.LBB0_124:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v96, v12, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v97, 31, v96
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[96:97], v[96:97], 1, v[80:81]
	v_perm_b32 v94, v243, v244, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[96:97], v[94:95], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_65
.LBB0_125:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v94, v12, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v95, 31, v94
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[94:95], v[94:95], 1, v[80:81]
	v_perm_b32 v92, v237, v242, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[94:95], v[92:93], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_66
.LBB0_126:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v92, v12, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v93, 31, v92
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[92:93], v[92:93], 1, v[80:81]
	v_perm_b32 v90, v235, v236, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[92:93], v[90:91], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_67
.LBB0_127:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v90, v12, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v91, 31, v90
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[90:91], v[90:91], 1, v[80:81]
	v_perm_b32 v88, v233, v234, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[90:91], v[88:89], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_68
.LBB0_128:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v88, v12, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v89, 31, v88
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[88:89], v[88:89], 1, v[80:81]
	v_perm_b32 v86, v231, v232, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[88:89], v[86:87], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_69
.LBB0_129:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v86, v12, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v87, 31, v86
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[86:87], v[86:87], 1, v[80:81]
	v_perm_b32 v84, v229, v230, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[86:87], v[84:85], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_70
	s_branch .LBB0_71
.LBB0_130:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v82, v12, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v83, 31, v82
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[80:81]
	v_perm_b32 v78, v34, v226, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[82:83], v[78:79], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_73
.LBB0_131:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v78, v12, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v79, 31, v78
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[78:79], v[78:79], 1, v[80:81]
	v_perm_b32 v76, v30, v32, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[78:79], v[76:77], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_74
.LBB0_132:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v76, v12, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v77, 31, v76
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[76:77], v[76:77], 1, v[80:81]
	v_perm_b32 v74, v46, v42, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[76:77], v[74:75], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_75
.LBB0_133:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v74, v12, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v75, 31, v74
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[74:75], v[74:75], 1, v[80:81]
	v_perm_b32 v72, v50, v44, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[74:75], v[72:73], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_76
.LBB0_134:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v72, v12, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v73, 31, v72
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[72:73], v[72:73], 1, v[80:81]
	v_perm_b32 v70, v54, v48, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[72:73], v[70:71], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_77
.LBB0_135:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v70, v12, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v71, 31, v70
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[80:81]
	v_perm_b32 v68, v58, v52, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[70:71], v[68:69], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_78
.LBB0_136:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v68, v12, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v69, 31, v68
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[80:81]
	v_perm_b32 v66, v62, v56, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[68:69], v[66:67], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_79
	s_branch .LBB0_80
.LBB0_137:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v64, v12, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v65, 31, v64
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[80:81]
	v_perm_b32 v62, v148, v150, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[64:65], v[62:63], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_82
.LBB0_138:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v62, v12, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v63, 31, v62
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[62:63], v[62:63], 1, v[80:81]
	v_perm_b32 v60, v144, v147, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[62:63], v[60:61], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_83
.LBB0_139:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v60, v12, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v61, 31, v60
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[60:61], v[60:61], 1, v[80:81]
	v_perm_b32 v58, v136, v143, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[60:61], v[58:59], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_84
.LBB0_140:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v58, v12, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v59, 31, v58
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[58:59], v[58:59], 1, v[80:81]
	v_perm_b32 v56, v40, v133, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[58:59], v[56:57], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_85
.LBB0_141:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v56, v12, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v57, 31, v56
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[56:57], v[56:57], 1, v[80:81]
	v_perm_b32 v54, v26, v28, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[56:57], v[54:55], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_86
.LBB0_142:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v54, v12, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v55, 31, v54
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[54:55], v[54:55], 1, v[80:81]
	v_perm_b32 v52, v20, v24, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[54:55], v[52:53], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_87
.LBB0_143:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v52, v12, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v53, 31, v52
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[52:53], v[52:53], 1, v[80:81]
	v_perm_b32 v50, v131, v130, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[52:53], v[50:51], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_88
	s_branch .LBB0_89
.LBB0_144:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v48, v12, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v49, 31, v48
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[48:49], v[48:49], 1, v[80:81]
	v_perm_b32 v46, v137, v138, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[48:49], v[46:47], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_91
.LBB0_145:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v46, v12, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v47, 31, v46
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[46:47], v[46:47], 1, v[80:81]
	v_perm_b32 v44, v139, v142, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[46:47], v[44:45], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_92
.LBB0_146:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v44, v12, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v45, 31, v44
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[44:45], v[44:45], 1, v[80:81]
	v_perm_b32 v42, v145, v146, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[44:45], v[42:43], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_93
.LBB0_147:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v42, v12, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v43, 31, v42
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[42:43], v[42:43], 1, v[80:81]
	v_perm_b32 v40, v151, v149, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[42:43], v[40:41], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_94
.LBB0_148:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v40, v12, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v41, 31, v40
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[40:41], v[40:41], 1, v[80:81]
	v_perm_b32 v38, v155, v153, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[40:41], v[38:39], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_95
.LBB0_149:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v38, v12, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v39, 31, v38
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[80:81]
	v_perm_b32 v36, v159, v156, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[38:39], v[36:37], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_96
.LBB0_150:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v36, v12, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v37, 31, v36
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[36:37], v[36:37], 1, v[80:81]
	v_perm_b32 v34, v163, v161, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[36:37], v[34:35], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_97
	s_branch .LBB0_98
.LBB0_151:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v32, v12, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v33, 31, v32
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[32:33], v[32:33], 1, v[80:81]
	v_perm_b32 v30, v171, v167, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[32:33], v[30:31], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_100
.LBB0_152:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v30, v12, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v31, 31, v30
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[30:31], v[30:31], 1, v[80:81]
	v_perm_b32 v28, v175, v170, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[30:31], v[28:29], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_101
.LBB0_153:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v28, v12, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v29, 31, v28
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[28:29], v[28:29], 1, v[80:81]
	v_perm_b32 v26, v174, v178, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[28:29], v[26:27], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_102
.LBB0_154:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v26, v12, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v27, 31, v26
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[26:27], v[26:27], 1, v[80:81]
	v_perm_b32 v24, v179, v182, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[26:27], v[24:25], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_103
.LBB0_155:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v24, v12, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v25, 31, v24
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[24:25], v[24:25], 1, v[80:81]
	v_perm_b32 v22, v16, v184, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[24:25], v[22:23], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_104
.LBB0_156:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v22, v12, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v23, 31, v22
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[22:23], v[22:23], 1, v[80:81]
	v_perm_b32 v20, v188, v189, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[22:23], v[20:21], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[22:23], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_105
.LBB0_157:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v20, v12, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v21, 31, v20
	s_mov_b32 s9, 0x5040100
	v_lshl_add_u64 v[20:21], v[20:21], 1, v[80:81]
	v_perm_b32 v18, v195, v190, s9
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[20:21], v[18:19], off
	s_or_b64 exec, exec, s[20:21]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[20:21], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[14:15], s[20:21]
	s_cbranch_execnz .LBB0_106
	s_branch .LBB0_107
.LBB0_158:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v18, v16, v2
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v19, 31, v18
	s_mov_b32 s12, 0x5040100
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[80:81]
	v_perm_b32 v14, v198, v199, s12
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[18:19], v[14:15], off
	s_or_b64 exec, exec, s[8:9]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[8:9], s[14:15], s[0:1]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB0_109
.LBB0_159:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v14, v16, v6
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v15, 31, v14
	s_mov_b32 s8, 0x5040100
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[80:81]
	v_perm_b32 v12, v206, v205, s8
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[14:15], v[12:13], off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[2:3]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_110
.LBB0_160:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v12, v16, v8
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v13, 31, v12
	s_mov_b32 s2, 0x5040100
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[80:81]
	v_perm_b32 v184, v204, v210, s2
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[12:13], v[184:185], off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[4:5]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_111
.LBB0_161:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v12, v16, v10
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v13, 31, v12
	s_mov_b32 s2, 0x5040100
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[80:81]
	v_perm_b32 v8, v215, v207, s2
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[12:13], v[8:9], off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[6:7]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_112
.LBB0_162:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v8, v16, v0
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v9, 31, v8
	s_mov_b32 s2, 0x5040100
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[80:81]
	v_perm_b32 v6, v213, v214, s2
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[8:9], v[6:7], off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[18:19]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_113
.LBB0_163:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v0, v16, v1
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s2, 0x5040100
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[80:81]
	v_perm_b32 v4, v220, v221, s2
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[0:1], v[4:5], off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[2:3], s[14:15], s[10:11]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_114
.LBB0_164:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v0, v16, v11
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s2, 0x5040100
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[80:81]
	v_perm_b32 v2, v132, v135, s2
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[0:1], v[2:3], off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 14                         ; 00_naive_mfma.py:90:14
	s_and_b64 s[0:1], s[14:15], s[16:17]
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_115
.LBB0_165:
	.loc	1 88 17                         ; 00_naive_mfma.py:88:17
	v_add_u32_e32 v0, v16, v116
	.loc	1 89 14                         ; 00_naive_mfma.py:89:14
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s0, 0x5040100
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[80:81]
	v_perm_b32 v182, v239, v238, s0
	.loc	1 91 5                          ; 00_naive_mfma.py:91:5
	global_store_dwordx2 v[0:1], v[182:183], off
	.loc	1 8 1                           ; 00_naive_mfma.py:8:1
	s_endpgm
.Ltmp13:
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
		.amdhsa_next_free_vgpr 444
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 256
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
	.set .Lv0_naive.num_vgpr, 256
	.set .Lv0_naive.num_agpr, 188
	.set .Lv0_naive.numbered_sgpr, 24
	.set .Lv0_naive.num_named_barrier, 0
	.set .Lv0_naive.private_seg_size, 0
	.set .Lv0_naive.uses_vcc, 1
	.set .Lv0_naive.uses_flat_scratch, 0
	.set .Lv0_naive.has_dyn_sized_stack, 0
	.set .Lv0_naive.has_recursion, 0
	.set .Lv0_naive.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14272
; TotalNumSgprs: 30
; NumVgprs: 256
; NumAgprs: 188
; TotalNumVgprs: 444
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 55
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 444
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x58 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x32 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	27                              ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	70                              ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"00_naive_mfma.py"              ; string offset=7 ; 00_naive_mfma.py
.Linfo_string2:
	.asciz	"/root"                         ; string offset=24 ; /root
.Linfo_string3:
	.asciz	"v0_naive"                      ; string offset=30 ; v0_naive
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     188
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
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         v0_naive.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     444
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
