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
; %bb.262:
	.file	1 "/root" "00_naive_mfma.py"
	.loc	1 8 0 prologue_end              ; 00_naive_mfma.py:8:0
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.263:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
.Ltmp1:
	.file	2 "/root/.venv/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 43 13 is_stmt 1               ; standard.py:43:13 @[ 00_naive_mfma.py:28:17 ]
	s_add_i32 s0, s15, 0xff
	.loc	2 43 12 is_stmt 0               ; standard.py:43:12 @[ 00_naive_mfma.py:28:17 ]
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 24
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 8
.Ltmp2:
	.loc	1 30 13 is_stmt 1               ; 00_naive_mfma.py:30:13
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_mov_b64 s[18:19], s[4:5]
	s_load_dword s82, s[4:5], 0x20
	s_sub_i32 s5, 0, s1
	v_rcp_f32_e32 v1, v1
	s_abs_i32 s3, s16
	v_and_b32_e32 v104, 0x3ff, v0
                                        ; implicit-def: $vgpr255 : SGPR spill to VGPR lane
	v_mov_b32_e32 v152, 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_readfirstlane_b32 s2, v104
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	s_bfe_u32 s4, s2, 0x20006
	v_mov_b32_e32 v212, 0
	.loc	1 30 13                         ; 00_naive_mfma.py:30:13
	v_readfirstlane_b32 s6, v1
	s_mul_i32 s5, s5, s6
	s_mul_hi_u32 s5, s6, s5
	s_add_i32 s6, s6, s5
	s_mul_hi_u32 s5, s3, s6
	s_mul_i32 s6, s5, s1
	v_writelane_b32 v255, s2, 0
	s_xor_b32 s2, s16, s0
	s_sub_i32 s3, s3, s6
	s_ashr_i32 s2, s2, 31
	s_add_i32 s6, s5, 1
	s_sub_i32 s7, s3, s1
	s_cmp_ge_u32 s3, s1
	s_cselect_b32 s5, s6, s5
	s_cselect_b32 s3, s7, s3
	s_add_i32 s6, s5, 1
	s_cmp_ge_u32 s3, s1
	s_cselect_b32 s1, s6, s5
	s_xor_b32 s1, s1, s2
	s_sub_i32 s1, s1, s2
	.loc	1 31 13                         ; 00_naive_mfma.py:31:13
	s_mul_i32 s0, s1, s0
	s_sub_i32 s0, s16, s0
	.loc	1 55 15                         ; 00_naive_mfma.py:55:15
	v_and_b32_e32 v1, 63, v0
	.loc	1 61 22                         ; 00_naive_mfma.py:61:22
	s_lshl_b32 s33, s1, 8
	.loc	1 62 22                         ; 00_naive_mfma.py:62:22
	s_lshl_b32 s76, s0, 8
.Ltmp3:
	.loc	2 43 13                         ; standard.py:43:13 @[ 00_naive_mfma.py:76:23 ]
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s82, 63
.Ltmp4:
	.loc	1 55 15                         ; 00_naive_mfma.py:55:15
	v_lshl_or_b32 v58, s4, 6, v1
	.loc	1 76 5                          ; 00_naive_mfma.py:76:5
	s_cmp_lt_i32 s2, 64
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_mov_b32_e32 v8, 0
	v_accvgpr_write_b32 a19, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v22, 0
	v_accvgpr_write_b32 a20, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v14, 0
	v_accvgpr_write_b32 a21, 0
	v_mov_b32_e32 v12, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v74, 0
	v_accvgpr_write_b32 a24, 0
	v_mov_b32_e32 v76, 0
	v_accvgpr_write_b32 a25, 0
	v_mov_b32_e32 v78, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v253, 0
	s_cbranch_scc1 .LBB0_259
; %bb.1:                                ; %.lr.ph
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	v_writelane_b32 v255, s12, 1
	.loc	1 55 15 is_stmt 1               ; 00_naive_mfma.py:55:15
	v_lshrrev_b32_e32 v0, 5, v58
	.loc	1 56 15                         ; 00_naive_mfma.py:56:15
	v_lshlrev_b32_e32 v2, 1, v104
	v_writelane_b32 v255, s13, 2
	v_writelane_b32 v255, s18, 3
	s_load_dwordx2 s[0:1], s[18:19], 0x24
	v_accvgpr_write_b32 a1, v58
	v_and_b32_e32 v97, 62, v2
	.loc	1 55 15                         ; 00_naive_mfma.py:55:15
	v_or_b32_e32 v2, 0xf8, v0
	v_or_b32_e32 v4, 0xf0, v0
	v_or_b32_e32 v6, 0xe8, v0
	v_or_b32_e32 v8, 0xe0, v0
	v_or_b32_e32 v10, 0xd8, v0
	v_or_b32_e32 v12, 0xd0, v0
	v_or_b32_e32 v14, 0xc8, v0
	v_or_b32_e32 v16, 0xc0, v0
	v_or_b32_e32 v18, 0xb8, v0
	v_or_b32_e32 v20, 0xb0, v0
	v_or_b32_e32 v22, 0xa8, v0
	v_or_b32_e32 v24, 0xa0, v0
	v_or_b32_e32 v26, 0x98, v0
	v_or_b32_e32 v28, 0x90, v0
	v_or_b32_e32 v30, 0x88, v0
	v_or_b32_e32 v32, 0x80, v0
	v_or_b32_e32 v34, 0x78, v0
	v_or_b32_e32 v36, 0x70, v0
	v_or_b32_e32 v38, 0x68, v0
	v_or_b32_e32 v40, 0x60, v0
	v_or_b32_e32 v42, 0x58, v0
	v_or_b32_e32 v44, 0x50, v0
	v_or_b32_e32 v46, 0x48, v0
	v_or_b32_e32 v48, 64, v0
	v_or_b32_e32 v50, 56, v0
	v_or_b32_e32 v52, 48, v0
	v_or_b32_e32 v54, 40, v0
	v_or_b32_e32 v56, 32, v0
	v_or_b32_e32 v58, 24, v0
	v_or_b32_e32 v60, 16, v0
	v_or_b32_e32 v62, 8, v0
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v64, v0, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v80, v0, s0
	v_or_b32_e32 v81, s33, v0
	v_or_b32_e32 v0, s76, v0
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v63, v62, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v62, v62, s0
	v_cmp_gt_i32_e64 s[36:37], s15, v0
	v_add_lshl_u32 v0, v80, v97, 1
	v_mul_lo_u32 v79, v60, s0
	v_accvgpr_write_b32 a3, v0
	v_add_lshl_u32 v0, v62, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v59, v58, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v58, v58, s0
	v_accvgpr_write_b32 a4, v0
	v_add_lshl_u32 v0, v79, v97, 1
	v_mul_lo_u32 v78, v56, s0
	v_accvgpr_write_b32 a5, v0
	v_add_lshl_u32 v0, v58, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v55, v54, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v54, v54, s0
	v_accvgpr_write_b32 a6, v0
	v_add_lshl_u32 v0, v78, v97, 1
	v_mul_lo_u32 v77, v52, s0
	v_accvgpr_write_b32 a7, v0
	v_add_lshl_u32 v0, v54, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v51, v50, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v50, v50, s0
	v_accvgpr_write_b32 a8, v0
	v_add_lshl_u32 v0, v77, v97, 1
	v_mul_lo_u32 v76, v48, s0
	v_accvgpr_write_b32 a9, v0
	v_add_lshl_u32 v0, v50, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v47, v46, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v46, v46, s0
	v_accvgpr_write_b32 a10, v0
	v_add_lshl_u32 v0, v76, v97, 1
	v_mul_lo_u32 v75, v44, s0
	v_accvgpr_write_b32 a11, v0
	v_add_lshl_u32 v0, v46, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v43, v42, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v42, v42, s0
	v_accvgpr_write_b32 a12, v0
	v_add_lshl_u32 v0, v75, v97, 1
	v_mul_lo_u32 v74, v40, s0
	v_accvgpr_write_b32 a13, v0
	v_add_lshl_u32 v0, v42, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v39, v38, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v38, v38, s0
	v_accvgpr_write_b32 a14, v0
	v_add_lshl_u32 v0, v74, v97, 1
	v_mul_lo_u32 v73, v36, s0
	v_accvgpr_write_b32 a15, v0
	v_add_lshl_u32 v0, v38, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v35, v34, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v34, v34, s0
	v_accvgpr_write_b32 a16, v0
	v_add_lshl_u32 v0, v73, v97, 1
	v_mul_lo_u32 v72, v32, s0
	v_accvgpr_write_b32 a17, v0
	v_add_lshl_u32 v0, v34, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v31, v30, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v30, v30, s0
	v_accvgpr_write_b32 a18, v0
	v_add_lshl_u32 v0, v72, v97, 1
	v_mul_lo_u32 v71, v28, s0
	v_accvgpr_write_b32 a19, v0
	v_add_lshl_u32 v0, v30, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v27, v26, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v26, v26, s0
	v_accvgpr_write_b32 a20, v0
	v_add_lshl_u32 v0, v71, v97, 1
	v_mul_lo_u32 v70, v24, s0
	v_accvgpr_write_b32 a21, v0
	v_add_lshl_u32 v0, v26, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v23, v22, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v22, v22, s0
	v_accvgpr_write_b32 a22, v0
	v_add_lshl_u32 v0, v70, v97, 1
	v_mul_lo_u32 v69, v20, s0
	v_accvgpr_write_b32 a23, v0
	v_add_lshl_u32 v0, v22, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v19, v18, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v18, v18, s0
	v_accvgpr_write_b32 a24, v0
	v_add_lshl_u32 v0, v69, v97, 1
	v_mul_lo_u32 v68, v16, s0
	v_accvgpr_write_b32 a25, v0
	v_add_lshl_u32 v0, v18, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v15, v14, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v14, v14, s0
	v_accvgpr_write_b32 a26, v0
	v_add_lshl_u32 v0, v68, v97, 1
	v_mul_lo_u32 v67, v12, s0
	v_accvgpr_write_b32 a27, v0
	v_add_lshl_u32 v0, v14, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v11, v10, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v10, v10, s0
	v_accvgpr_write_b32 a28, v0
	v_add_lshl_u32 v0, v67, v97, 1
	v_mul_lo_u32 v66, v8, s0
	v_accvgpr_write_b32 a29, v0
	v_add_lshl_u32 v0, v10, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v7, v6, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v6, v6, s0
	v_accvgpr_write_b32 a30, v0
	v_add_lshl_u32 v0, v66, v97, 1
	v_mul_lo_u32 v65, v4, s0
	v_accvgpr_write_b32 a31, v0
	v_add_lshl_u32 v0, v6, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v3, v2, s1
	.loc	1 64 17                         ; 00_naive_mfma.py:64:17
	v_mul_lo_u32 v2, v2, s0
	v_accvgpr_write_b32 a32, v0
	v_add_lshl_u32 v0, v65, v97, 1
	v_accvgpr_write_b32 a33, v0
	v_add_lshl_u32 v0, v2, v97, 1
	v_accvgpr_write_b32 a34, v0
	v_add_lshl_u32 v0, v64, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v61, v60, s1
	v_accvgpr_write_b32 a35, v0
	v_add_lshl_u32 v0, v63, v97, 1
	v_accvgpr_write_b32 a36, v0
	v_add_lshl_u32 v0, v61, v97, 1
	v_mul_lo_u32 v57, v56, s1
	v_accvgpr_write_b32 a37, v0
	v_add_lshl_u32 v0, v59, v97, 1
	v_accvgpr_write_b32 a38, v0
	v_add_lshl_u32 v0, v57, v97, 1
	v_mul_lo_u32 v53, v52, s1
	v_accvgpr_write_b32 a39, v0
	v_add_lshl_u32 v0, v55, v97, 1
	v_accvgpr_write_b32 a40, v0
	v_add_lshl_u32 v0, v53, v97, 1
.Ltmp5:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:76:23 ]
	s_ashr_i32 s3, s2, 31
.Ltmp6:
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v49, v48, s1
	v_accvgpr_write_b32 a41, v0
	v_add_lshl_u32 v0, v51, v97, 1
.Ltmp7:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:76:23 ]
	s_lshr_b32 s3, s3, 26
	v_accvgpr_write_b32 a42, v0
	v_add_lshl_u32 v0, v49, v97, 1
	s_add_i32 s2, s2, s3
.Ltmp8:
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v45, v44, s1
	v_accvgpr_write_b32 a43, v0
	v_add_lshl_u32 v0, v47, v97, 1
.Ltmp9:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:76:23 ]
	s_ashr_i32 s83, s2, 6
.Ltmp10:
	.loc	1 62 22                         ; 00_naive_mfma.py:62:22
	s_mul_i32 s2, s76, s1
	v_accvgpr_write_b32 a44, v0
	v_add_lshl_u32 v0, v45, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v41, v40, s1
	.loc	1 62 14                         ; 00_naive_mfma.py:62:14
	s_ashr_i32 s3, s2, 31
	v_accvgpr_write_b32 a45, v0
	v_add_lshl_u32 v0, v43, v97, 1
	s_lshl_b64 s[2:3], s[2:3], 1
	v_accvgpr_write_b32 a46, v0
	v_add_lshl_u32 v0, v41, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v37, v36, s1
	.loc	1 62 14                         ; 00_naive_mfma.py:62:14
	s_add_u32 s84, s10, s2
	.loc	1 61 22                         ; 00_naive_mfma.py:61:22
	s_mul_i32 s0, s33, s0
	v_accvgpr_write_b32 a47, v0
	v_add_lshl_u32 v0, v39, v97, 1
	.loc	1 65 48                         ; 00_naive_mfma.py:65:48
	v_mul_lo_u32 v5, v4, s1
	v_mul_lo_u32 v9, v8, s1
	v_mul_lo_u32 v13, v12, s1
	v_mul_lo_u32 v17, v16, s1
	v_mul_lo_u32 v21, v20, s1
	v_mul_lo_u32 v25, v24, s1
	v_mul_lo_u32 v29, v28, s1
	v_mul_lo_u32 v33, v32, s1
	.loc	1 62 14                         ; 00_naive_mfma.py:62:14
	s_addc_u32 s85, s11, s3
	.loc	1 61 14                         ; 00_naive_mfma.py:61:14
	s_ashr_i32 s1, s0, 31
	v_accvgpr_write_b32 a48, v0
	v_add_lshl_u32 v0, v37, v97, 1
	s_lshl_b64 s[0:1], s[0:1], 1
	v_accvgpr_write_b32 a49, v0
	v_add_lshl_u32 v0, v35, v97, 1
	s_add_u32 s86, s8, s0
	v_accvgpr_write_b32 a50, v0
	v_add_lshl_u32 v0, v33, v97, 1
	s_addc_u32 s87, s9, s1
	s_lshl_b32 s88, s4, 8
	v_accvgpr_write_b32 a51, v0
	v_add_lshl_u32 v0, v31, v97, 1
	v_writelane_b32 v255, s19, 4
	s_add_i32 s5, s88, 0xb000
	v_accvgpr_write_b32 a52, v0
	v_add_lshl_u32 v0, v29, v97, 1
	v_writelane_b32 v255, s5, 5
	s_add_i32 s5, s88, 0xb400
	v_accvgpr_write_b32 a53, v0
	v_add_lshl_u32 v0, v27, v97, 1
	v_writelane_b32 v255, s5, 6
	s_add_i32 s5, s88, 0xb800
	v_accvgpr_write_b32 a54, v0
	v_add_lshl_u32 v0, v25, v97, 1
	v_writelane_b32 v255, s5, 7
	s_add_i32 s5, s88, 0xbc00
	v_accvgpr_write_b32 a55, v0
	v_add_lshl_u32 v0, v23, v97, 1
	v_writelane_b32 v255, s5, 8
	s_add_i32 s5, s88, 0xc000
	v_accvgpr_write_b32 a56, v0
	v_add_lshl_u32 v0, v21, v97, 1
	v_writelane_b32 v255, s5, 9
	s_add_i32 s5, s88, 0xc400
	v_accvgpr_write_b32 a57, v0
	v_add_lshl_u32 v0, v19, v97, 1
	v_writelane_b32 v255, s5, 10
	s_add_i32 s5, s88, 0xc800
	v_accvgpr_write_b32 a58, v0
	v_add_lshl_u32 v0, v17, v97, 1
	v_writelane_b32 v255, s5, 11
	s_add_i32 s5, s88, 0xcc00
	v_accvgpr_write_b32 a59, v0
	v_add_lshl_u32 v0, v15, v97, 1
	v_writelane_b32 v255, s5, 12
	s_add_i32 s5, s88, 0xd000
	v_accvgpr_write_b32 a60, v0
	v_add_lshl_u32 v0, v13, v97, 1
	v_writelane_b32 v255, s5, 13
	s_add_i32 s5, s88, 0xd400
	v_accvgpr_write_b32 a61, v0
	v_add_lshl_u32 v0, v11, v97, 1
	v_writelane_b32 v255, s5, 14
	s_add_i32 s5, s88, 0xd800
	v_accvgpr_write_b32 a62, v0
	v_add_lshl_u32 v0, v9, v97, 1
	v_writelane_b32 v255, s5, 15
	s_add_i32 s5, s88, 0xdc00
	v_accvgpr_write_b32 a63, v0
	v_add_lshl_u32 v0, v7, v97, 1
	v_writelane_b32 v255, s5, 16
	s_add_i32 s5, s88, 0xe000
	v_accvgpr_write_b32 a64, v0
	v_add_lshl_u32 v0, v5, v97, 1
	v_lshl_add_u32 v254, v1, 2, s88
	v_writelane_b32 v255, s5, 17
	s_add_i32 s5, s88, 0xe400
	v_accvgpr_write_b32 a65, v0
	v_add_lshl_u32 v0, v3, v97, 1
	v_lshrrev_b32_e32 v1, 1, v104
	v_writelane_b32 v255, s5, 18
	s_add_i32 s5, s88, 0xe800
	v_accvgpr_write_b32 a66, v0
	v_lshlrev_b32_e32 v0, 7, v104
	v_and_b32_e32 v1, 24, v1
	s_movk_i32 s12, 0x780
	v_writelane_b32 v255, s5, 19
	v_and_or_b32 v0, v0, s12, v1
	s_add_i32 s12, s88, 0xec00
	v_writelane_b32 v255, s12, 20
	s_add_i32 s12, s88, 0xf000
	v_writelane_b32 v255, s12, 21
	s_add_i32 s12, s88, 0xf800
	v_writelane_b32 v255, s12, 22
	s_add_i32 s12, s88, 0xfc00
	v_writelane_b32 v255, s12, 23
	s_add_i32 s12, s88, 0xf400
	v_writelane_b32 v255, s12, 24
	s_add_i32 s12, s88, 0xac00
	v_writelane_b32 v255, s12, 25
	s_add_i32 s12, s88, 0xa800
	v_writelane_b32 v255, s12, 26
	s_add_i32 s12, s88, 0xa400
	v_writelane_b32 v255, s12, 27
	s_add_i32 s12, s88, 0xa000
	v_writelane_b32 v255, s12, 28
	s_add_i32 s12, s88, 0x9c00
	v_writelane_b32 v255, s12, 29
	s_add_i32 s12, s88, 0x9800
	v_writelane_b32 v255, s12, 30
	s_add_i32 s12, s88, 0x9400
	v_writelane_b32 v255, s12, 31
	s_add_i32 s12, s88, 0x9000
	v_writelane_b32 v255, s12, 32
	s_add_i32 s12, s88, 0x8c00
	v_writelane_b32 v255, s12, 33
	s_add_i32 s12, s88, 0x8800
	v_writelane_b32 v255, s12, 34
	s_add_i32 s12, s88, 0x8400
	v_writelane_b32 v255, s12, 35
	s_add_i32 s12, s88, 0x8000
	v_writelane_b32 v255, s12, 36
	s_add_i32 s12, s88, 0x7c00
	v_writelane_b32 v255, s12, 37
	s_add_i32 s12, s88, 0x7800
	v_writelane_b32 v255, s12, 38
	s_add_i32 s12, s88, 0x7400
	v_writelane_b32 v255, s12, 39
	s_add_i32 s12, s88, 0x7000
	v_writelane_b32 v255, s12, 40
	s_add_i32 s12, s88, 0x6c00
	v_writelane_b32 v255, s12, 41
	s_add_i32 s12, s88, 0x6800
	v_writelane_b32 v255, s12, 42
	s_add_i32 s12, s88, 0x6400
	v_writelane_b32 v255, s12, 43
	s_add_i32 s12, s88, 0x6000
	v_writelane_b32 v255, s12, 44
	s_add_i32 s12, s88, 0x5c00
	v_writelane_b32 v255, s12, 45
	s_add_i32 s12, s88, 0x5800
	v_writelane_b32 v255, s12, 46
	s_add_i32 s12, s88, 0x5400
	v_writelane_b32 v255, s12, 47
	s_add_i32 s12, s88, 0x5000
	v_writelane_b32 v255, s12, 48
	s_add_i32 s12, s88, 0x4c00
	v_writelane_b32 v255, s12, 49
	s_add_i32 s12, s88, 0x4800
	v_writelane_b32 v255, s12, 50
	s_add_i32 s12, s88, 0x4400
	v_writelane_b32 v255, s12, 51
	s_add_i32 s12, s88, 0x4000
	v_mov_b32_e32 v148, 0
	v_writelane_b32 v255, s12, 52
	s_add_i32 s12, s88, 0x3c00
	v_mov_b32_e32 v3, v148
	v_writelane_b32 v255, s12, 53
	s_add_i32 s12, s88, 0x3800
	v_or_b32_e32 v82, s33, v60
	v_or_b32_e32 v83, s33, v56
	v_or_b32_e32 v84, s33, v52
	v_or_b32_e32 v85, s33, v48
	v_or_b32_e32 v86, s33, v44
	v_or_b32_e32 v87, s33, v40
	v_or_b32_e32 v88, s33, v36
	v_or_b32_e32 v89, s33, v32
	v_or_b32_e32 v90, s33, v28
	v_or_b32_e32 v91, s33, v24
	v_or_b32_e32 v92, s33, v20
	v_or_b32_e32 v93, s33, v16
	v_or_b32_e32 v94, s33, v12
	v_or_b32_e32 v95, s33, v8
	v_or_b32_e32 v96, s33, v4
	v_or_b32_e32 v60, s76, v60
	v_or_b32_e32 v56, s76, v56
	v_or_b32_e32 v52, s76, v52
	v_or_b32_e32 v48, s76, v48
	v_or_b32_e32 v44, s76, v44
	v_or_b32_e32 v40, s76, v40
	v_or_b32_e32 v36, s76, v36
	v_or_b32_e32 v32, s76, v32
	v_or_b32_e32 v28, s76, v28
	v_or_b32_e32 v24, s76, v24
	v_or_b32_e32 v20, s76, v20
	v_or_b32_e32 v16, s76, v16
	v_or_b32_e32 v12, s76, v12
	v_or_b32_e32 v8, s76, v8
	v_or_b32_e32 v4, s76, v4
	s_lshl_b32 s5, s4, 10
	v_accvgpr_write_b32 a67, v0
	s_lshl_b32 s4, s4, 11
	v_mov_b32_e32 v0, v148
	v_mov_b32_e32 v1, v148
	v_mov_b32_e32 v2, v148
	v_accvgpr_write_b32 a155, v3
	v_accvgpr_write_b32 a151, v3
	v_accvgpr_write_b32 a147, v3
	v_accvgpr_write_b32 a143, v3
	v_accvgpr_write_b32 a139, v3
	v_accvgpr_write_b32 a135, v3
	v_accvgpr_write_b32 a131, v3
	v_accvgpr_write_b32 a127, v3
	v_accvgpr_write_b32 a123, v3
	v_accvgpr_write_b32 a119, v3
	v_accvgpr_write_b32 a115, v3
	v_accvgpr_write_b32 a111, v3
	v_accvgpr_write_b32 a107, v3
	v_accvgpr_write_b32 a103, v3
	v_accvgpr_write_b32 a99, v3
	v_accvgpr_write_b32 a95, v3
	v_accvgpr_write_b32 a79, v3
	v_accvgpr_write_b32 a71, v3
	v_accvgpr_write_b32 a75, v3
	v_accvgpr_write_b32 a83, v3
	v_writelane_b32 v255, s12, 54
	s_add_i32 s12, s88, 0x3400
	v_accvgpr_write_b32 a91, v3
	v_accvgpr_write_b32 a87, v3
	v_cmp_gt_i32_e32 vcc, s14, v81
	v_cmp_gt_i32_e64 s[0:1], s14, v82
	v_cmp_gt_i32_e64 s[2:3], s14, v83
	v_cmp_gt_i32_e64 s[6:7], s14, v84
	v_cmp_gt_i32_e64 s[8:9], s14, v85
	v_cmp_gt_i32_e64 s[10:11], s14, v86
	v_cmp_gt_i32_e64 s[70:71], s14, v87
	v_cmp_gt_i32_e64 s[16:17], s14, v88
	v_cmp_gt_i32_e64 s[18:19], s14, v89
	v_cmp_gt_i32_e64 s[20:21], s14, v90
	v_cmp_gt_i32_e64 s[22:23], s14, v91
	v_cmp_gt_i32_e64 s[24:25], s14, v92
	v_cmp_gt_i32_e64 s[26:27], s14, v93
	v_cmp_gt_i32_e64 s[28:29], s14, v94
	v_cmp_gt_i32_e64 s[30:31], s14, v95
	v_cmp_gt_i32_e64 s[34:35], s14, v96
	v_cmp_gt_i32_e64 s[38:39], s15, v60
	v_cmp_gt_i32_e64 s[40:41], s15, v56
	v_cmp_gt_i32_e64 s[42:43], s15, v52
	v_cmp_gt_i32_e64 s[44:45], s15, v48
	v_cmp_gt_i32_e64 s[46:47], s15, v44
	v_cmp_gt_i32_e64 s[48:49], s15, v40
	v_cmp_gt_i32_e64 s[50:51], s15, v36
	v_cmp_gt_i32_e64 s[52:53], s15, v32
	v_cmp_gt_i32_e64 s[54:55], s15, v28
	v_cmp_gt_i32_e64 s[56:57], s15, v24
	v_cmp_gt_i32_e64 s[58:59], s15, v20
	v_cmp_gt_i32_e64 s[60:61], s15, v16
	v_accvgpr_write_b32 a2, v97
	v_accvgpr_write_b32 a0, v104
	s_and_b32 s5, s5, 0x800
	s_and_b32 s4, s4, 0x800
	s_mov_b32 s75, 0x27000
	s_mov_b32 s74, 0x7ffffffe
	v_mov_b32_e32 v149, v148
	v_mov_b32_e32 v150, v148
	v_mov_b32_e32 v151, v148
	v_accvgpr_write_b32 a154, v2
	v_accvgpr_write_b32 a153, v1
	v_accvgpr_write_b32 a152, v0
	v_accvgpr_write_b32 a150, v2
	v_accvgpr_write_b32 a149, v1
	v_accvgpr_write_b32 a148, v0
	v_accvgpr_write_b32 a146, v2
	v_accvgpr_write_b32 a145, v1
	v_accvgpr_write_b32 a144, v0
	v_accvgpr_write_b32 a142, v2
	v_accvgpr_write_b32 a141, v1
	v_accvgpr_write_b32 a140, v0
	v_accvgpr_write_b32 a138, v2
	v_accvgpr_write_b32 a137, v1
	v_accvgpr_write_b32 a136, v0
	v_accvgpr_write_b32 a134, v2
	v_accvgpr_write_b32 a133, v1
	v_accvgpr_write_b32 a132, v0
	v_accvgpr_write_b32 a130, v2
	v_accvgpr_write_b32 a129, v1
	v_accvgpr_write_b32 a128, v0
	v_accvgpr_write_b32 a126, v2
	v_accvgpr_write_b32 a125, v1
	v_accvgpr_write_b32 a124, v0
	v_accvgpr_write_b32 a122, v2
	v_accvgpr_write_b32 a121, v1
	v_accvgpr_write_b32 a120, v0
	v_accvgpr_write_b32 a118, v2
	v_accvgpr_write_b32 a117, v1
	v_accvgpr_write_b32 a116, v0
	v_accvgpr_write_b32 a114, v2
	v_accvgpr_write_b32 a113, v1
	v_accvgpr_write_b32 a112, v0
	v_accvgpr_write_b32 a110, v2
	v_accvgpr_write_b32 a109, v1
	v_accvgpr_write_b32 a108, v0
	v_accvgpr_write_b32 a106, v2
	v_accvgpr_write_b32 a105, v1
	v_accvgpr_write_b32 a104, v0
	v_accvgpr_write_b32 a102, v2
	v_accvgpr_write_b32 a101, v1
	v_accvgpr_write_b32 a100, v0
	v_accvgpr_write_b32 a98, v2
	v_accvgpr_write_b32 a97, v1
	v_accvgpr_write_b32 a96, v0
	v_mov_b32_e32 v200, v148
	v_mov_b32_e32 v201, v148
	v_mov_b32_e32 v202, v148
	v_mov_b32_e32 v203, v148
	v_mov_b32_e32 v196, v148
	v_mov_b32_e32 v197, v148
	v_mov_b32_e32 v198, v148
	v_mov_b32_e32 v199, v148
	v_mov_b32_e32 v192, v148
	v_mov_b32_e32 v193, v148
	v_mov_b32_e32 v194, v148
	v_mov_b32_e32 v195, v148
	v_mov_b32_e32 v188, v148
	v_mov_b32_e32 v189, v148
	v_mov_b32_e32 v190, v148
	v_mov_b32_e32 v191, v148
	v_mov_b32_e32 v184, v148
	v_mov_b32_e32 v185, v148
	v_mov_b32_e32 v186, v148
	v_mov_b32_e32 v187, v148
	v_mov_b32_e32 v180, v148
	v_mov_b32_e32 v181, v148
	v_mov_b32_e32 v182, v148
	v_mov_b32_e32 v183, v148
	v_mov_b32_e32 v176, v148
	v_mov_b32_e32 v177, v148
	v_mov_b32_e32 v178, v148
	v_mov_b32_e32 v179, v148
	v_accvgpr_write_b32 a94, v2
	v_accvgpr_write_b32 a93, v1
	v_accvgpr_write_b32 a92, v0
	v_mov_b32_e32 v168, v148
	v_mov_b32_e32 v169, v148
	v_mov_b32_e32 v170, v148
	v_mov_b32_e32 v171, v148
	v_mov_b32_e32 v164, v148
	v_mov_b32_e32 v165, v148
	v_mov_b32_e32 v166, v148
	v_mov_b32_e32 v167, v148
	v_mov_b32_e32 v248, v148
	v_mov_b32_e32 v249, v148
	v_mov_b32_e32 v250, v148
	v_mov_b32_e32 v251, v148
	v_mov_b32_e32 v144, v148
	v_mov_b32_e32 v145, v148
	v_mov_b32_e32 v146, v148
	v_mov_b32_e32 v147, v148
	v_mov_b32_e32 v140, v148
	v_mov_b32_e32 v141, v148
	v_mov_b32_e32 v142, v148
	v_mov_b32_e32 v143, v148
	v_mov_b32_e32 v136, v148
	v_mov_b32_e32 v137, v148
	v_mov_b32_e32 v138, v148
	v_mov_b32_e32 v139, v148
	v_mov_b32_e32 v132, v148
	v_mov_b32_e32 v133, v148
	v_mov_b32_e32 v134, v148
	v_mov_b32_e32 v135, v148
	v_mov_b32_e32 v128, v148
	v_mov_b32_e32 v129, v148
	v_mov_b32_e32 v130, v148
	v_mov_b32_e32 v131, v148
	v_mov_b32_e32 v124, v148
	v_mov_b32_e32 v125, v148
	v_mov_b32_e32 v126, v148
	v_mov_b32_e32 v127, v148
	v_mov_b32_e32 v104, v148
	v_mov_b32_e32 v105, v148
	v_mov_b32_e32 v106, v148
	v_mov_b32_e32 v107, v148
	v_mov_b32_e32 v92, v148
	v_mov_b32_e32 v93, v148
	v_mov_b32_e32 v94, v148
	v_mov_b32_e32 v95, v148
	v_mov_b32_e32 v80, v148
	v_mov_b32_e32 v81, v148
	v_mov_b32_e32 v82, v148
	v_mov_b32_e32 v83, v148
	v_mov_b32_e32 v68, v148
	v_mov_b32_e32 v69, v148
	v_mov_b32_e32 v70, v148
	v_mov_b32_e32 v71, v148
	v_mov_b32_e32 v56, v148
	v_mov_b32_e32 v57, v148
	v_mov_b32_e32 v58, v148
	v_mov_b32_e32 v59, v148
	v_mov_b32_e32 v44, v148
	v_mov_b32_e32 v45, v148
	v_mov_b32_e32 v46, v148
	v_mov_b32_e32 v47, v148
	v_accvgpr_write_b32 a78, v2
	v_accvgpr_write_b32 a77, v1
	v_accvgpr_write_b32 a76, v0
	v_accvgpr_write_b32 a70, v2
	v_accvgpr_write_b32 a69, v1
	v_accvgpr_write_b32 a68, v0
	v_mov_b32_e32 v48, v148
	v_mov_b32_e32 v49, v148
	v_mov_b32_e32 v50, v148
	v_mov_b32_e32 v51, v148
	v_mov_b32_e32 v52, v148
	v_mov_b32_e32 v53, v148
	v_mov_b32_e32 v54, v148
	v_mov_b32_e32 v55, v148
	v_mov_b32_e32 v60, v148
	v_mov_b32_e32 v61, v148
	v_mov_b32_e32 v62, v148
	v_mov_b32_e32 v63, v148
	v_mov_b32_e32 v64, v148
	v_mov_b32_e32 v65, v148
	v_mov_b32_e32 v66, v148
	v_mov_b32_e32 v67, v148
	v_cmp_gt_i32_e64 s[62:63], s15, v12
	v_cmp_gt_i32_e64 s[64:65], s15, v8
	v_cmp_gt_i32_e64 s[66:67], s15, v4
	v_mov_b32_e32 v72, v148
	v_mov_b32_e32 v73, v148
	v_mov_b32_e32 v74, v148
	v_mov_b32_e32 v75, v148
	v_mov_b32_e32 v76, v148
	v_mov_b32_e32 v77, v148
	v_mov_b32_e32 v78, v148
	v_mov_b32_e32 v79, v148
	v_accvgpr_write_b32 a74, v2
	v_accvgpr_write_b32 a73, v1
	v_accvgpr_write_b32 a72, v0
	v_accvgpr_write_b32 a82, v2
	v_accvgpr_write_b32 a81, v1
	v_accvgpr_write_b32 a80, v0
	v_mov_b32_e32 v96, v148
	v_mov_b32_e32 v97, v148
	v_mov_b32_e32 v98, v148
	v_mov_b32_e32 v99, v148
	v_mov_b32_e32 v100, v148
	v_mov_b32_e32 v101, v148
	v_mov_b32_e32 v102, v148
	v_mov_b32_e32 v103, v148
	v_mov_b32_e32 v108, v148
	v_mov_b32_e32 v109, v148
	v_mov_b32_e32 v110, v148
	v_mov_b32_e32 v111, v148
	v_mov_b32_e32 v112, v148
	v_mov_b32_e32 v113, v148
	v_mov_b32_e32 v114, v148
	v_mov_b32_e32 v115, v148
	v_mov_b32_e32 v118, v148
	v_mov_b32_e32 v119, v148
	v_mov_b32_e32 v120, v148
	v_mov_b32_e32 v121, v148
	v_writelane_b32 v255, s12, 55
	v_mov_b32_e32 v152, v148
	v_mov_b32_e32 v153, v148
	v_mov_b32_e32 v154, v148
	v_mov_b32_e32 v155, v148
	s_add_i32 s98, s88, 0x3000
	s_add_i32 s99, s88, 0x2c00
	s_add_i32 s77, s88, 0x2800
	s_add_i32 s13, s88, 0x2400
	v_accvgpr_write_b32 a90, v2
	v_accvgpr_write_b32 a89, v1
	v_accvgpr_write_b32 a88, v0
	s_add_i32 s12, s88, 0x2000
	s_add_i32 s89, s88, 0x1c00
	s_add_i32 s90, s88, 0x1800
	s_add_i32 s91, s88, 0x1400
	v_accvgpr_write_b32 a86, v2
	v_accvgpr_write_b32 a85, v1
	v_accvgpr_write_b32 a84, v0
	s_add_i32 s92, s88, 0x1000
	s_add_i32 s93, s88, 0xc00
	s_add_i32 s94, s88, 0x800
	s_add_i32 s95, s88, 0x400
	v_mov_b32_e32 v236, v148
	v_mov_b32_e32 v237, v148
	v_mov_b32_e32 v238, v148
	v_mov_b32_e32 v239, v148
	v_mov_b32_e32 v232, v148
	v_mov_b32_e32 v233, v148
	v_mov_b32_e32 v234, v148
	v_mov_b32_e32 v235, v148
	v_mov_b32_e32 v228, v148
	v_mov_b32_e32 v229, v148
	v_mov_b32_e32 v230, v148
	v_mov_b32_e32 v231, v148
	v_mov_b32_e32 v224, v148
	v_mov_b32_e32 v225, v148
	v_mov_b32_e32 v226, v148
	v_mov_b32_e32 v227, v148
	v_mov_b32_e32 v220, v148
	v_mov_b32_e32 v221, v148
	v_mov_b32_e32 v222, v148
	v_mov_b32_e32 v223, v148
	v_mov_b32_e32 v216, v148
	v_mov_b32_e32 v217, v148
	v_mov_b32_e32 v218, v148
	v_mov_b32_e32 v219, v148
	v_mov_b32_e32 v212, v148
	v_mov_b32_e32 v213, v148
	v_mov_b32_e32 v214, v148
	v_mov_b32_e32 v215, v148
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 14 is_stmt 0                ; 00_naive_mfma.py:0:14
	s_or_b64 exec, exec, s[68:69]
	v_accvgpr_read_b32 v0, a67
	.loc	1 86 17 is_stmt 1               ; 00_naive_mfma.py:86:17
	v_add_u32_e32 v36, s4, v0
	v_add_u32_e32 v162, 0x8000, v36
	; asyncmark
	; wait_asyncmark(0)
	.loc	1 84 13                         ; 00_naive_mfma.py:84:13
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	v_add_u32_e32 v252, s5, v0
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	ds_read2_b64 v[4:7], v162 offset1:4
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[0:3], v252 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[16:19], v[4:5], v[0:1], v[148:151]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	v_add_u32_e32 v156, 0x9000, v36
	v_add_u32_e32 v208, 0xa000, v36
	v_add_u32_e32 v209, 0xb000, v36
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[16:19], v[6:7], v[2:3], v[16:19]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	v_add_u32_e32 v163, 0xc000, v36
	v_add_u32_e32 v210, 0xd000, v36
	v_add_u32_e32 v122, 0xe000, v36
	v_add_u32_e32 v123, 0xf000, v36
	ds_read2_b64 v[8:11], v156 offset1:4
	ds_read2_b64 v[12:15], v208 offset1:4
	ds_read2_b64 v[20:23], v163 offset1:4
	v_accvgpr_write_b32 a159, v19
	v_accvgpr_write_b32 a158, v18
	v_accvgpr_write_b32 a157, v17
	v_accvgpr_write_b32 a156, v16
	ds_read2_b64 v[16:19], v209 offset1:4
	ds_read2_b64 v[24:27], v210 offset1:4
	ds_read2_b64 v[28:31], v122 offset1:4
	ds_read2_b64 v[32:35], v123 offset1:4
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	v_add_u32_e32 v116, 0x1000, v252
	ds_read2_b64 v[36:39], v116 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[124:127], v[4:5], v[36:37], a[124:127]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	v_add_u32_e32 v157, 0x2000, v252
	v_add_u32_e32 v246, 0x3000, v252
	v_add_u32_e32 v117, 0x6000, v252
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[120:123], v[8:9], v[36:37], a[120:123]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	v_add_u32_e32 v245, 0x4000, v252
	v_add_u32_e32 v247, 0x5000, v252
	v_add_u32_e32 v244, 0x7000, v252
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[116:119], v[12:13], v[36:37], a[116:119]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[84:87], v247 offset1:4
	v_accvgpr_read_b32 v148, a156
	v_accvgpr_read_b32 v149, a157
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[112:115], v[16:17], v[36:37], a[112:115]
	v_accvgpr_read_b32 v150, a158
	v_accvgpr_read_b32 v151, a159
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[172:175], v246 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[108:111], v[20:21], v[36:37], a[108:111]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[204:207], v157 offset0:8 offset1:12
	.loc	1 96 9                          ; 00_naive_mfma.py:96:9
	s_add_u32 s86, s86, 0x80
	s_addc_u32 s87, s87, 0
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[104:107], v[24:25], v[36:37], a[104:107]
	.loc	1 97 9                          ; 00_naive_mfma.py:97:9
	s_add_u32 s84, s84, 0x80
	s_addc_u32 s85, s85, 0
	.loc	1 76 5                          ; 00_naive_mfma.py:76:5
	s_add_i32 s83, s83, -1
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[100:103], v[28:29], v[36:37], a[100:103]
	.loc	1 76 5                          ; 00_naive_mfma.py:76:5
	s_sub_i32 s82, s82, 64
	s_cmp_lg_u32 s83, 0
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[96:99], v[32:33], v[36:37], a[96:99]
	v_mfma_f32_16x16x16_f16 a[124:127], v[6:7], v[38:39], a[124:127]
	v_mfma_f32_16x16x16_f16 a[120:123], v[10:11], v[38:39], a[120:123]
	v_mfma_f32_16x16x16_f16 a[116:119], v[14:15], v[38:39], a[116:119]
	v_mfma_f32_16x16x16_f16 a[112:115], v[18:19], v[38:39], a[112:115]
	v_mfma_f32_16x16x16_f16 a[108:111], v[22:23], v[38:39], a[108:111]
	v_mfma_f32_16x16x16_f16 a[104:107], v[26:27], v[38:39], a[104:107]
	v_mfma_f32_16x16x16_f16 a[100:103], v[30:31], v[38:39], a[100:103]
	v_mfma_f32_16x16x16_f16 a[96:99], v[34:35], v[38:39], a[96:99]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[36:39], v157 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[40:43], v[4:5], v[36:37], v[200:203]
	v_mfma_f32_16x16x16_f16 v[200:203], v[6:7], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[8:9], v[36:37], v[196:199]
	v_mfma_f32_16x16x16_f16 v[196:199], v[10:11], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[12:13], v[36:37], v[192:195]
	v_mfma_f32_16x16x16_f16 v[192:195], v[14:15], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[16:17], v[36:37], v[188:191]
	v_mfma_f32_16x16x16_f16 v[188:191], v[18:19], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[20:21], v[36:37], v[184:187]
	v_mfma_f32_16x16x16_f16 v[184:187], v[22:23], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[24:25], v[36:37], v[180:183]
	v_mfma_f32_16x16x16_f16 v[180:183], v[26:27], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[28:29], v[36:37], v[176:179]
	v_mfma_f32_16x16x16_f16 a[92:95], v[32:33], v[36:37], a[92:95]
	v_mfma_f32_16x16x16_f16 v[176:179], v[30:31], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 a[92:95], v[34:35], v[38:39], a[92:95]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[36:39], v246 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[40:43], v[4:5], v[36:37], v[168:171]
	v_mfma_f32_16x16x16_f16 v[168:171], v[6:7], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[8:9], v[36:37], v[164:167]
	v_mfma_f32_16x16x16_f16 v[164:167], v[10:11], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[12:13], v[36:37], v[248:251]
	v_mfma_f32_16x16x16_f16 v[158:161], v[14:15], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[16:17], v[36:37], v[144:147]
	v_mfma_f32_16x16x16_f16 v[144:147], v[18:19], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[20:21], v[36:37], v[140:143]
	v_mfma_f32_16x16x16_f16 a[152:155], v[8:9], v[0:1], a[152:155]
	v_mfma_f32_16x16x16_f16 a[148:151], v[12:13], v[0:1], a[148:151]
	v_mfma_f32_16x16x16_f16 a[144:147], v[16:17], v[0:1], a[144:147]
	v_mfma_f32_16x16x16_f16 a[140:143], v[20:21], v[0:1], a[140:143]
	v_mfma_f32_16x16x16_f16 a[136:139], v[24:25], v[0:1], a[136:139]
	v_mfma_f32_16x16x16_f16 a[132:135], v[28:29], v[0:1], a[132:135]
	v_mfma_f32_16x16x16_f16 a[128:131], v[32:33], v[0:1], a[128:131]
	v_mfma_f32_16x16x16_f16 v[140:143], v[22:23], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[24:25], v[36:37], v[136:139]
	v_mfma_f32_16x16x16_f16 a[152:155], v[10:11], v[2:3], a[152:155]
	v_mfma_f32_16x16x16_f16 a[148:151], v[14:15], v[2:3], a[148:151]
	v_mfma_f32_16x16x16_f16 a[144:147], v[18:19], v[2:3], a[144:147]
	v_mfma_f32_16x16x16_f16 a[140:143], v[22:23], v[2:3], a[140:143]
	v_mfma_f32_16x16x16_f16 a[136:139], v[26:27], v[2:3], a[136:139]
	v_mfma_f32_16x16x16_f16 a[132:135], v[30:31], v[2:3], a[132:135]
	v_mfma_f32_16x16x16_f16 a[128:131], v[34:35], v[2:3], a[128:131]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[0:3], v117 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[136:139], v[26:27], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[28:29], v[36:37], v[132:135]
	v_mfma_f32_16x16x16_f16 v[132:135], v[30:31], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[32:33], v[36:37], v[128:131]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[248:251], v[24:25], v[0:1], v[118:121]
	v_mfma_f32_16x16x16_f16 v[128:131], v[34:35], v[38:39], v[40:43]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[36:39], v245 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[118:121], v[26:27], v[2:3], v[248:251]
	v_mfma_f32_16x16x16_f16 v[248:251], v[28:29], v[0:1], v[152:155]
	v_mfma_f32_16x16x16_f16 v[152:155], v[30:31], v[2:3], v[248:251]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	s_nop 5
	ds_read2_b64 v[248:251], v244 offset1:4
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[40:43], v[4:5], v[36:37], v[124:127]
	v_mfma_f32_16x16x16_f16 v[124:127], v[6:7], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[8:9], v[36:37], v[104:107]
	v_mfma_f32_16x16x16_f16 v[48:51], v[8:9], v[84:85], v[48:51]
	v_mfma_f32_16x16x16_f16 v[96:99], v[8:9], v[0:1], v[96:99]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[240:243], v[8:9], v[248:249], v[236:239]
	v_mfma_f32_16x16x16_f16 v[104:107], v[10:11], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[12:13], v[36:37], v[92:95]
	v_mfma_f32_16x16x16_f16 v[48:51], v[10:11], v[86:87], v[48:51]
	v_mfma_f32_16x16x16_f16 v[52:55], v[12:13], v[84:85], v[52:55]
	v_mfma_f32_16x16x16_f16 v[96:99], v[10:11], v[2:3], v[96:99]
	v_mfma_f32_16x16x16_f16 v[100:103], v[12:13], v[0:1], v[100:103]
	v_mfma_f32_16x16x16_f16 v[8:11], v[10:11], v[250:251], v[240:243]
	v_mfma_f32_16x16x16_f16 v[240:243], v[12:13], v[248:249], v[232:235]
	v_mfma_f32_16x16x16_f16 v[92:95], v[14:15], v[38:39], v[40:43]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 1
	ds_read2_b64 v[232:235], v208 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[40:43], v[16:17], v[36:37], v[80:83]
	v_mfma_f32_16x16x16_f16 v[52:55], v[14:15], v[86:87], v[52:55]
	v_mfma_f32_16x16x16_f16 v[60:63], v[16:17], v[84:85], v[60:63]
	v_mfma_f32_16x16x16_f16 v[100:103], v[14:15], v[2:3], v[100:103]
	v_mfma_f32_16x16x16_f16 v[108:111], v[16:17], v[0:1], v[108:111]
	v_mfma_f32_16x16x16_f16 v[240:243], v[14:15], v[250:251], v[240:243]
	v_mfma_f32_16x16x16_f16 v[12:15], v[16:17], v[248:249], v[228:231]
	v_mfma_f32_16x16x16_f16 v[80:83], v[18:19], v[38:39], v[40:43]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 1
	ds_read2_b64 v[228:231], v209 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[40:43], v[20:21], v[36:37], v[68:71]
	v_mfma_f32_16x16x16_f16 v[60:63], v[18:19], v[86:87], v[60:63]
	v_mfma_f32_16x16x16_f16 v[64:67], v[20:21], v[84:85], v[64:67]
	v_mfma_f32_16x16x16_f16 v[108:111], v[18:19], v[2:3], v[108:111]
	v_mfma_f32_16x16x16_f16 v[112:115], v[20:21], v[0:1], v[112:115]
	v_mfma_f32_16x16x16_f16 v[16:19], v[18:19], v[250:251], v[12:15]
	v_mfma_f32_16x16x16_f16 v[12:15], v[20:21], v[248:249], v[224:227]
	v_mfma_f32_16x16x16_f16 v[68:71], v[22:23], v[38:39], v[40:43]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 1
	ds_read2_b64 v[224:227], v163 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[40:43], v[24:25], v[36:37], v[56:59]
	v_mfma_f32_16x16x16_f16 v[64:67], v[22:23], v[86:87], v[64:67]
	v_mfma_f32_16x16x16_f16 v[72:75], v[24:25], v[84:85], v[72:75]
	v_mfma_f32_16x16x16_f16 v[112:115], v[22:23], v[2:3], v[112:115]
	v_mfma_f32_16x16x16_f16 v[20:23], v[22:23], v[250:251], v[12:15]
	v_mfma_f32_16x16x16_f16 v[12:15], v[24:25], v[248:249], v[220:223]
	v_mfma_f32_16x16x16_f16 v[56:59], v[26:27], v[38:39], v[40:43]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 1
	ds_read2_b64 v[220:223], v210 offset0:8 offset1:12
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[208:211], v116 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 v[40:43], v[28:29], v[36:37], v[44:47]
	v_mfma_f32_16x16x16_f16 v[72:75], v[26:27], v[86:87], v[72:75]
	v_mfma_f32_16x16x16_f16 v[76:79], v[28:29], v[84:85], v[76:79]
	v_mfma_f32_16x16x16_f16 v[24:27], v[26:27], v[250:251], v[12:15]
	v_mfma_f32_16x16x16_f16 v[12:15], v[28:29], v[248:249], v[216:219]
	v_mfma_f32_16x16x16_f16 v[44:47], v[30:31], v[38:39], v[40:43]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 1
	ds_read2_b64 v[216:219], v122 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[76:79], v[32:33], v[36:37], a[76:79]
	v_mfma_f32_16x16x16_f16 v[76:79], v[30:31], v[86:87], v[76:79]
	v_mfma_f32_16x16x16_f16 a[72:75], v[32:33], v[84:85], a[72:75]
	v_mfma_f32_16x16x16_f16 a[88:91], v[32:33], v[0:1], a[88:91]
	v_mfma_f32_16x16x16_f16 a[84:87], v[4:5], v[248:249], a[84:87]
	v_mfma_f32_16x16x16_f16 v[28:31], v[30:31], v[250:251], v[12:15]
	v_mfma_f32_16x16x16_f16 v[12:15], v[32:33], v[248:249], v[212:215]
	v_mfma_f32_16x16x16_f16 a[76:79], v[34:35], v[38:39], a[76:79]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[36:39], v247 offset0:8 offset1:12
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 0
	ds_read2_b64 v[212:215], v123 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[72:75], v[34:35], v[86:87], a[72:75]
	v_mfma_f32_16x16x16_f16 a[88:91], v[34:35], v[2:3], a[88:91]
	v_mfma_f32_16x16x16_f16 a[84:87], v[6:7], v[250:251], a[84:87]
	v_mfma_f32_16x16x16_f16 v[32:35], v[34:35], v[250:251], v[12:15]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 2
	ds_read2_b64 v[12:15], v162 offset0:8 offset1:12
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[248:251], v252 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[236:239], v[12:13], v[248:249], v[148:151]
	v_mfma_f32_16x16x16_f16 v[148:151], v[14:15], v[250:251], v[236:239]
	.loc	1 86 17                         ; 00_naive_mfma.py:86:17
	s_nop 5
	ds_read2_b64 v[236:239], v156 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[40:43], v[236:237], v[36:37], v[48:51]
	v_mfma_f32_16x16x16_f16 v[48:51], v[238:239], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[232:233], v[36:37], v[52:55]
	v_mfma_f32_16x16x16_f16 v[52:55], v[234:235], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[228:229], v[36:37], v[60:63]
	v_mfma_f32_16x16x16_f16 a[68:71], v[4:5], v[84:85], a[68:71]
	v_mfma_f32_16x16x16_f16 v[60:63], v[230:231], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[224:225], v[36:37], v[64:67]
	v_mfma_f32_16x16x16_f16 a[68:71], v[6:7], v[86:87], a[68:71]
	v_mfma_f32_16x16x16_f16 v[64:67], v[226:227], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[220:221], v[36:37], v[72:75]
	v_mfma_f32_16x16x16_f16 a[80:83], v[4:5], v[0:1], a[80:83]
	v_mfma_f32_16x16x16_f16 a[152:155], v[236:237], v[248:249], a[152:155]
	v_mfma_f32_16x16x16_f16 a[148:151], v[232:233], v[248:249], a[148:151]
	v_mfma_f32_16x16x16_f16 a[144:147], v[228:229], v[248:249], a[144:147]
	v_mfma_f32_16x16x16_f16 a[140:143], v[224:225], v[248:249], a[140:143]
	v_mfma_f32_16x16x16_f16 a[136:139], v[220:221], v[248:249], a[136:139]
	v_mfma_f32_16x16x16_f16 a[132:135], v[216:217], v[248:249], a[132:135]
	v_mfma_f32_16x16x16_f16 a[128:131], v[212:213], v[248:249], a[128:131]
	v_mfma_f32_16x16x16_f16 v[156:159], v[232:233], v[172:173], v[158:161]
	v_mfma_f32_16x16x16_f16 a[68:71], v[12:13], v[36:37], a[68:71]
	v_mfma_f32_16x16x16_f16 v[72:75], v[222:223], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[216:217], v[36:37], v[76:79]
	v_mfma_f32_16x16x16_f16 a[72:75], v[212:213], v[36:37], a[72:75]
	v_mfma_f32_16x16x16_f16 a[80:83], v[6:7], v[2:3], a[80:83]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[0:3], v244 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[152:155], v[238:239], v[250:251], a[152:155]
	v_mfma_f32_16x16x16_f16 a[148:151], v[234:235], v[250:251], a[148:151]
	v_mfma_f32_16x16x16_f16 a[144:147], v[230:231], v[250:251], a[144:147]
	v_mfma_f32_16x16x16_f16 a[140:143], v[226:227], v[250:251], a[140:143]
	v_mfma_f32_16x16x16_f16 a[136:139], v[222:223], v[250:251], a[136:139]
	v_mfma_f32_16x16x16_f16 a[132:135], v[218:219], v[250:251], a[132:135]
	v_mfma_f32_16x16x16_f16 a[128:131], v[214:215], v[250:251], a[128:131]
	v_mfma_f32_16x16x16_f16 v[248:251], v[234:235], v[174:175], v[156:159]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	s_nop 2
	ds_read2_b64 v[156:159], v245 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[68:71], v[14:15], v[38:39], a[68:71]
	v_mfma_f32_16x16x16_f16 v[76:79], v[218:219], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 a[72:75], v[214:215], v[38:39], a[72:75]
	.loc	1 85 17                         ; 00_naive_mfma.py:85:17
	ds_read2_b64 v[36:39], v117 offset0:8 offset1:12
	.loc	1 93 15                         ; 00_naive_mfma.py:93:15
	v_mfma_f32_16x16x16_f16 a[120:123], v[236:237], v[208:209], a[120:123]
	v_mfma_f32_16x16x16_f16 v[196:199], v[236:237], v[204:205], v[196:199]
	v_mfma_f32_16x16x16_f16 v[162:165], v[236:237], v[172:173], v[164:167]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[104:107], v[236:237], v[156:157], v[104:107]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[40:43], v[236:237], v[36:37], v[96:99]
	v_mfma_f32_16x16x16_f16 v[4:7], v[236:237], v[0:1], v[8:11]
	v_mfma_f32_16x16x16_f16 a[120:123], v[238:239], v[210:211], a[120:123]
	v_mfma_f32_16x16x16_f16 a[116:119], v[232:233], v[208:209], a[116:119]
	v_mfma_f32_16x16x16_f16 v[196:199], v[238:239], v[206:207], v[196:199]
	v_mfma_f32_16x16x16_f16 v[192:195], v[232:233], v[204:205], v[192:195]
	v_mfma_f32_16x16x16_f16 v[164:167], v[238:239], v[174:175], v[162:165]
	v_mfma_f32_16x16x16_f16 v[104:107], v[238:239], v[158:159], v[104:107]
	v_mfma_f32_16x16x16_f16 v[92:95], v[232:233], v[156:157], v[92:95]
	v_mfma_f32_16x16x16_f16 v[96:99], v[238:239], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[232:233], v[36:37], v[100:103]
	v_mfma_f32_16x16x16_f16 v[236:239], v[238:239], v[2:3], v[4:7]
	v_mfma_f32_16x16x16_f16 v[4:7], v[232:233], v[0:1], v[240:243]
	v_mfma_f32_16x16x16_f16 a[116:119], v[234:235], v[210:211], a[116:119]
	v_mfma_f32_16x16x16_f16 a[112:115], v[228:229], v[208:209], a[112:115]
	v_mfma_f32_16x16x16_f16 v[192:195], v[234:235], v[206:207], v[192:195]
	v_mfma_f32_16x16x16_f16 v[188:191], v[228:229], v[204:205], v[188:191]
	v_mfma_f32_16x16x16_f16 v[144:147], v[228:229], v[172:173], v[144:147]
	v_mfma_f32_16x16x16_f16 v[92:95], v[234:235], v[158:159], v[92:95]
	v_mfma_f32_16x16x16_f16 v[80:83], v[228:229], v[156:157], v[80:83]
	v_mfma_f32_16x16x16_f16 v[100:103], v[234:235], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[228:229], v[36:37], v[108:111]
	v_mfma_f32_16x16x16_f16 v[232:235], v[234:235], v[2:3], v[4:7]
	v_mfma_f32_16x16x16_f16 v[4:7], v[228:229], v[0:1], v[16:19]
	v_mfma_f32_16x16x16_f16 a[112:115], v[230:231], v[210:211], a[112:115]
	v_mfma_f32_16x16x16_f16 a[108:111], v[224:225], v[208:209], a[108:111]
	v_mfma_f32_16x16x16_f16 v[188:191], v[230:231], v[206:207], v[188:191]
	v_mfma_f32_16x16x16_f16 v[184:187], v[224:225], v[204:205], v[184:187]
	v_mfma_f32_16x16x16_f16 v[144:147], v[230:231], v[174:175], v[144:147]
	v_mfma_f32_16x16x16_f16 v[140:143], v[224:225], v[172:173], v[140:143]
	v_mfma_f32_16x16x16_f16 v[80:83], v[230:231], v[158:159], v[80:83]
	v_mfma_f32_16x16x16_f16 v[68:71], v[224:225], v[156:157], v[68:71]
	v_mfma_f32_16x16x16_f16 v[108:111], v[230:231], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[224:225], v[36:37], v[112:115]
	v_mfma_f32_16x16x16_f16 v[228:231], v[230:231], v[2:3], v[4:7]
	v_mfma_f32_16x16x16_f16 v[4:7], v[224:225], v[0:1], v[20:23]
	v_mfma_f32_16x16x16_f16 a[108:111], v[226:227], v[210:211], a[108:111]
	v_mfma_f32_16x16x16_f16 a[104:107], v[220:221], v[208:209], a[104:107]
	v_mfma_f32_16x16x16_f16 v[184:187], v[226:227], v[206:207], v[184:187]
	v_mfma_f32_16x16x16_f16 v[180:183], v[220:221], v[204:205], v[180:183]
	v_mfma_f32_16x16x16_f16 v[140:143], v[226:227], v[174:175], v[140:143]
	v_mfma_f32_16x16x16_f16 v[136:139], v[220:221], v[172:173], v[136:139]
	v_mfma_f32_16x16x16_f16 v[68:71], v[226:227], v[158:159], v[68:71]
	v_mfma_f32_16x16x16_f16 v[56:59], v[220:221], v[156:157], v[56:59]
	v_mfma_f32_16x16x16_f16 v[112:115], v[226:227], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[220:221], v[36:37], v[118:121]
	v_mfma_f32_16x16x16_f16 v[224:227], v[226:227], v[2:3], v[4:7]
	v_mfma_f32_16x16x16_f16 v[4:7], v[220:221], v[0:1], v[24:27]
	v_mfma_f32_16x16x16_f16 a[104:107], v[222:223], v[210:211], a[104:107]
	v_mfma_f32_16x16x16_f16 a[100:103], v[216:217], v[208:209], a[100:103]
	v_mfma_f32_16x16x16_f16 v[180:183], v[222:223], v[206:207], v[180:183]
	v_mfma_f32_16x16x16_f16 v[176:179], v[216:217], v[204:205], v[176:179]
	v_mfma_f32_16x16x16_f16 v[136:139], v[222:223], v[174:175], v[136:139]
	v_mfma_f32_16x16x16_f16 v[132:135], v[216:217], v[172:173], v[132:135]
	v_mfma_f32_16x16x16_f16 v[56:59], v[222:223], v[158:159], v[56:59]
	v_mfma_f32_16x16x16_f16 v[44:47], v[216:217], v[156:157], v[44:47]
	v_mfma_f32_16x16x16_f16 v[118:121], v[222:223], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 v[40:43], v[216:217], v[36:37], v[152:155]
	v_mfma_f32_16x16x16_f16 v[220:223], v[222:223], v[2:3], v[4:7]
	v_mfma_f32_16x16x16_f16 v[4:7], v[216:217], v[0:1], v[28:31]
	v_mfma_f32_16x16x16_f16 a[124:127], v[12:13], v[208:209], a[124:127]
	v_mfma_f32_16x16x16_f16 a[100:103], v[218:219], v[210:211], a[100:103]
	v_mfma_f32_16x16x16_f16 a[96:99], v[212:213], v[208:209], a[96:99]
	v_mfma_f32_16x16x16_f16 v[200:203], v[12:13], v[204:205], v[200:203]
	v_mfma_f32_16x16x16_f16 v[176:179], v[218:219], v[206:207], v[176:179]
	v_mfma_f32_16x16x16_f16 a[92:95], v[212:213], v[204:205], a[92:95]
	v_mfma_f32_16x16x16_f16 v[168:171], v[12:13], v[172:173], v[168:171]
	v_mfma_f32_16x16x16_f16 v[132:135], v[218:219], v[174:175], v[132:135]
	v_mfma_f32_16x16x16_f16 v[128:131], v[212:213], v[172:173], v[128:131]
	v_mfma_f32_16x16x16_f16 v[122:125], v[12:13], v[156:157], v[124:127]
	v_mfma_f32_16x16x16_f16 v[44:47], v[218:219], v[158:159], v[44:47]
	v_mfma_f32_16x16x16_f16 a[76:79], v[212:213], v[156:157], a[76:79]
	v_mfma_f32_16x16x16_f16 a[80:83], v[12:13], v[36:37], a[80:83]
	v_mfma_f32_16x16x16_f16 v[152:155], v[218:219], v[38:39], v[40:43]
	v_mfma_f32_16x16x16_f16 a[88:91], v[212:213], v[36:37], a[88:91]
	v_mfma_f32_16x16x16_f16 a[84:87], v[12:13], v[0:1], a[84:87]
	v_mfma_f32_16x16x16_f16 v[216:219], v[218:219], v[2:3], v[4:7]
	v_mfma_f32_16x16x16_f16 v[4:7], v[212:213], v[0:1], v[32:35]
	v_mfma_f32_16x16x16_f16 a[124:127], v[14:15], v[210:211], a[124:127]
	v_mfma_f32_16x16x16_f16 a[96:99], v[214:215], v[210:211], a[96:99]
	v_mfma_f32_16x16x16_f16 v[200:203], v[14:15], v[206:207], v[200:203]
	v_mfma_f32_16x16x16_f16 a[92:95], v[214:215], v[206:207], a[92:95]
	v_mfma_f32_16x16x16_f16 v[168:171], v[14:15], v[174:175], v[168:171]
	v_mfma_f32_16x16x16_f16 v[128:131], v[214:215], v[174:175], v[128:131]
	v_mfma_f32_16x16x16_f16 v[124:127], v[14:15], v[158:159], v[122:125]
	v_mfma_f32_16x16x16_f16 a[76:79], v[214:215], v[158:159], a[76:79]
	v_mfma_f32_16x16x16_f16 a[80:83], v[14:15], v[38:39], a[80:83]
	v_mfma_f32_16x16x16_f16 a[88:91], v[214:215], v[38:39], a[88:91]
	v_mfma_f32_16x16x16_f16 a[84:87], v[14:15], v[2:3], a[84:87]
	v_mfma_f32_16x16x16_f16 v[212:215], v[214:215], v[2:3], v[4:7]
	.loc	1 76 5                          ; 00_naive_mfma.py:76:5
	s_cbranch_scc0 .LBB0_260
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 0 5 is_stmt 0                 ; 00_naive_mfma.py:0:5
	v_accvgpr_read_b32 v0, a2
	.loc	1 77 62 is_stmt 1               ; 00_naive_mfma.py:77:62
	v_cmp_gt_i32_e64 s[68:69], s82, v0
	.loc	1 77 18 is_stmt 0               ; 00_naive_mfma.py:77:18
	s_and_b64 s[72:73], vcc, s[68:69]
	s_xor_b64 s[78:79], s[72:73], -1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_barrier
	s_and_saveexec_b64 s[72:73], s[78:79]
	s_xor_b64 s[72:73], exec, s[72:73]
	s_cbranch_execnz .LBB0_162
; %bb.4:                                ; %Flow792
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_saveexec_b64 s[80:81], s[72:73]
	s_and_b32 s73, s87, 0xffff
	s_xor_b64 exec, exec, s[80:81]
	s_cbranch_execnz .LBB0_163
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_164
.LBB0_6:                                ; %Flow791
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_8
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s95
	v_accvgpr_read_b32 v0, a4
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_8:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[0:1], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_165
; %bb.9:                                ; %Flow789
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_166
.LBB0_10:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_167
.LBB0_11:                               ; %Flow788
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_13
.LBB0_12:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s93
	v_accvgpr_read_b32 v0, a6
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[2:3], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_168
; %bb.14:                               ; %Flow786
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_169
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_170
.LBB0_16:                               ; %Flow785
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_18
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s91
	v_accvgpr_read_b32 v0, a8
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_18:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[6:7], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_171
; %bb.19:                               ; %Flow783
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_172
.LBB0_20:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_173
.LBB0_21:                               ; %Flow782
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_23
.LBB0_22:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s89
	v_accvgpr_read_b32 v0, a10
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_23:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[8:9], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_174
; %bb.24:                               ; %Flow780
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_175
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_176
.LBB0_26:                               ; %Flow779
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_28
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s13
	v_accvgpr_read_b32 v0, a12
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_28:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[10:11], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_177
; %bb.29:                               ; %Flow777
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_178
.LBB0_30:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_179
.LBB0_31:                               ; %Flow776
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_33
.LBB0_32:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s99
	v_accvgpr_read_b32 v0, a14
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[70:71], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_180
; %bb.34:                               ; %Flow774
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_181
.LBB0_35:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_182
.LBB0_36:                               ; %Flow773
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_38
.LBB0_37:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 55
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a16
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_38:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[16:17], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_183
; %bb.39:                               ; %Flow771
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_184
.LBB0_40:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_185
.LBB0_41:                               ; %Flow770
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_43
.LBB0_42:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 53
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a18
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_43:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[18:19], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_186
; %bb.44:                               ; %Flow768
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_187
.LBB0_45:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_188
.LBB0_46:                               ; %Flow767
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_48
.LBB0_47:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 51
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a20
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_48:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[20:21], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_189
; %bb.49:                               ; %Flow765
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_190
.LBB0_50:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_191
.LBB0_51:                               ; %Flow764
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_53
.LBB0_52:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 49
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a22
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_53:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[22:23], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_192
; %bb.54:                               ; %Flow762
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_193
.LBB0_55:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_194
.LBB0_56:                               ; %Flow761
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_58
.LBB0_57:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 47
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a24
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_58:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[24:25], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_195
; %bb.59:                               ; %Flow759
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_196
.LBB0_60:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_197
.LBB0_61:                               ; %Flow758
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_63
.LBB0_62:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 45
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a26
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_63:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[26:27], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_198
; %bb.64:                               ; %Flow756
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_199
.LBB0_65:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_200
.LBB0_66:                               ; %Flow755
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_68
.LBB0_67:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 43
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a28
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_68:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[28:29], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_201
; %bb.69:                               ; %Flow753
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_202
.LBB0_70:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_203
.LBB0_71:                               ; %Flow752
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_73
.LBB0_72:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 41
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a30
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_73:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[30:31], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_204
; %bb.74:                               ; %Flow750
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_205
.LBB0_75:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_206
.LBB0_76:                               ; %Flow749
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_78
.LBB0_77:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 39
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a32
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_78:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 77 18 is_stmt 1               ; 00_naive_mfma.py:77:18
	s_and_b64 s[78:79], s[34:35], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_207
; %bb.79:                               ; %Flow747
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_208
.LBB0_80:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_209
.LBB0_81:                               ; %Flow746
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_83
.LBB0_82:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	v_readlane_b32 s80, v255, 37
	s_mov_b32 s72, s86
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a34
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_83:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[72:73], s[36:37], s[68:69]
	s_xor_b64 s[78:79], s[72:73], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[72:73], s[78:79]
	s_xor_b64 s[72:73], exec, s[72:73]
	s_cbranch_execnz .LBB0_210
; %bb.84:                               ; %Flow744
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_saveexec_b64 s[80:81], s[72:73]
	s_and_b32 s73, s85, 0xffff
	s_xor_b64 exec, exec, s[80:81]
	s_cbranch_execnz .LBB0_211
.LBB0_85:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_212
.LBB0_86:                               ; %Flow743
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_88
.LBB0_87:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 35
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a36
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_88:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[38:39], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_213
; %bb.89:                               ; %Flow741
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_214
.LBB0_90:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_215
.LBB0_91:                               ; %Flow740
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_93
.LBB0_92:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 33
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a38
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_93:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[40:41], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_216
; %bb.94:                               ; %Flow738
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_217
.LBB0_95:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_218
.LBB0_96:                               ; %Flow737
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_98
.LBB0_97:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 31
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a40
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_98:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[42:43], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_219
; %bb.99:                               ; %Flow735
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_220
.LBB0_100:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_221
.LBB0_101:                              ; %Flow734
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_103
.LBB0_102:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 29
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a42
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_103:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[44:45], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_222
; %bb.104:                              ; %Flow732
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_223
.LBB0_105:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_224
.LBB0_106:                              ; %Flow731
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_108
.LBB0_107:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 27
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a44
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_108:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[46:47], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_225
; %bb.109:                              ; %Flow729
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_226
.LBB0_110:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_227
.LBB0_111:                              ; %Flow728
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_113
.LBB0_112:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 25
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a46
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_113:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[48:49], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_228
; %bb.114:                              ; %Flow726
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_229
.LBB0_115:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_230
.LBB0_116:                              ; %Flow725
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_118
.LBB0_117:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 6
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a48
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_118:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[50:51], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_231
; %bb.119:                              ; %Flow723
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_232
.LBB0_120:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_233
.LBB0_121:                              ; %Flow722
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_123
.LBB0_122:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 8
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a50
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_123:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[52:53], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_234
; %bb.124:                              ; %Flow720
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_235
.LBB0_125:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_236
.LBB0_126:                              ; %Flow719
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_128
.LBB0_127:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 10
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a52
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_128:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[54:55], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_237
; %bb.129:                              ; %Flow717
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_238
.LBB0_130:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_239
.LBB0_131:                              ; %Flow716
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_133
.LBB0_132:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 12
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a54
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_133:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[56:57], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_240
; %bb.134:                              ; %Flow714
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_241
.LBB0_135:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_242
.LBB0_136:                              ; %Flow713
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_138
.LBB0_137:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 14
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a56
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_138:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[58:59], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_243
; %bb.139:                              ; %Flow711
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_244
.LBB0_140:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_245
.LBB0_141:                              ; %Flow710
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_143
.LBB0_142:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 16
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a58
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_143:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[60:61], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_246
; %bb.144:                              ; %Flow708
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_247
.LBB0_145:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_248
.LBB0_146:                              ; %Flow707
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_148
.LBB0_147:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 18
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a60
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_148:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[62:63], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_249
; %bb.149:                              ; %Flow705
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_250
.LBB0_150:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_251
.LBB0_151:                              ; %Flow704
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_153
.LBB0_152:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 20
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a62
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_153:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[78:79], s[64:65], s[68:69]
	s_xor_b64 s[78:79], s[78:79], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[80:81], exec, s[80:81]
	s_cbranch_execnz .LBB0_252
; %bb.154:                              ; %Flow702
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execnz .LBB0_253
.LBB0_155:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[80:81]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execnz .LBB0_254
.LBB0_156:                              ; %Flow701
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_158
.LBB0_157:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 24
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a64
	buffer_load_dword v0, s[72:75], 0 offen lds
.LBB0_158:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_or_b64 exec, exec, s[78:79]
	.loc	1 78 18 is_stmt 1               ; 00_naive_mfma.py:78:18
	s_and_b64 s[68:69], s[66:67], s[68:69]
	s_xor_b64 s[68:69], s[68:69], -1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[78:79], s[68:69]
	s_xor_b64 s[78:79], exec, s[78:79]
	s_cbranch_execnz .LBB0_255
; %bb.159:                              ; %Flow699
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_256
.LBB0_160:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[78:79]
	.loc	1 82 13 is_stmt 1               ; 00_naive_mfma.py:82:13
	s_and_saveexec_b64 s[78:79], s[68:69]
	s_xor_b64 s[68:69], exec, s[78:79]
	s_cbranch_execnz .LBB0_257
.LBB0_161:                              ; %Flow
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	s_andn2_saveexec_b64 s[68:69], s[68:69]
	s_cbranch_execz .LBB0_2
	s_branch .LBB0_258
.LBB0_162:                              ; %.critedge
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13 is_stmt 1               ; 00_naive_mfma.py:81:13
	ds_write_b32 v254, v253
	s_or_saveexec_b64 s[80:81], s[72:73]
	s_and_b32 s73, s87, 0xffff
	s_xor_b64 exec, exec, s[80:81]
	s_cbranch_execz .LBB0_5
.LBB0_163:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s88
	v_accvgpr_read_b32 v0, a3
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_6
.LBB0_164:                              ; %.critedge2
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:1024
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_7
	s_branch .LBB0_8
.LBB0_165:                              ; %.critedge4
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:2048
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_10
.LBB0_166:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s94
	v_accvgpr_read_b32 v0, a5
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_11
.LBB0_167:                              ; %.critedge6
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:3072
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_12
	s_branch .LBB0_13
.LBB0_168:                              ; %.critedge8
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:4096
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_15
.LBB0_169:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s92
	v_accvgpr_read_b32 v0, a7
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_16
.LBB0_170:                              ; %.critedge10
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:5120
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_17
	s_branch .LBB0_18
.LBB0_171:                              ; %.critedge12
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:6144
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_20
.LBB0_172:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s90
	v_accvgpr_read_b32 v0, a9
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_21
.LBB0_173:                              ; %.critedge14
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:7168
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_22
	s_branch .LBB0_23
.LBB0_174:                              ; %.critedge16
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:8192
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_25
.LBB0_175:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s12
	v_accvgpr_read_b32 v0, a11
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_26
.LBB0_176:                              ; %.critedge18
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:9216
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_27
	s_branch .LBB0_28
.LBB0_177:                              ; %.critedge20
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:10240
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_30
.LBB0_178:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s77
	v_accvgpr_read_b32 v0, a13
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_31
.LBB0_179:                              ; %.critedge22
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:11264
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_32
	s_branch .LBB0_33
.LBB0_180:                              ; %.critedge24
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:12288
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_35
.LBB0_181:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_mov_b32 s72, s86
	s_mov_b32 m0, s98
	v_accvgpr_read_b32 v0, a15
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_36
.LBB0_182:                              ; %.critedge26
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:13312
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_37
	s_branch .LBB0_38
.LBB0_183:                              ; %.critedge28
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:14336
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_40
.LBB0_184:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 54
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a17
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_41
.LBB0_185:                              ; %.critedge30
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:15360
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_42
	s_branch .LBB0_43
.LBB0_186:                              ; %.critedge32
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:16384
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_45
.LBB0_187:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 52
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a19
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_46
.LBB0_188:                              ; %.critedge34
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:17408
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_47
	s_branch .LBB0_48
.LBB0_189:                              ; %.critedge36
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:18432
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_50
.LBB0_190:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 50
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a21
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_51
.LBB0_191:                              ; %.critedge38
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:19456
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_52
	s_branch .LBB0_53
.LBB0_192:                              ; %.critedge40
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:20480
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_55
.LBB0_193:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 48
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a23
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_56
.LBB0_194:                              ; %.critedge42
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:21504
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_57
	s_branch .LBB0_58
.LBB0_195:                              ; %.critedge44
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:22528
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_60
.LBB0_196:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 46
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a25
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_61
.LBB0_197:                              ; %.critedge46
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:23552
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_62
	s_branch .LBB0_63
.LBB0_198:                              ; %.critedge48
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:24576
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_65
.LBB0_199:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 44
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a27
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_66
.LBB0_200:                              ; %.critedge50
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:25600
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_67
	s_branch .LBB0_68
.LBB0_201:                              ; %.critedge52
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:26624
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_70
.LBB0_202:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 42
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a29
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_71
.LBB0_203:                              ; %.critedge54
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:27648
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_72
	s_branch .LBB0_73
.LBB0_204:                              ; %.critedge56
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:28672
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_75
.LBB0_205:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 40
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a31
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_76
.LBB0_206:                              ; %.critedge58
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:29696
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_77
	s_branch .LBB0_78
.LBB0_207:                              ; %.critedge60
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:30720
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_80
.LBB0_208:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	v_readlane_b32 s96, v255, 38
	s_mov_b32 s72, s86
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a33
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_81
.LBB0_209:                              ; %.critedge62
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:31744
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_82
	s_branch .LBB0_83
.LBB0_210:                              ; %.critedge64
                                        ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	ds_write_b32 v254, v253 offset:32768
	s_or_saveexec_b64 s[80:81], s[72:73]
	s_and_b32 s73, s85, 0xffff
	s_xor_b64 exec, exec, s[80:81]
	s_cbranch_execz .LBB0_85
.LBB0_211:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 36
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a35
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_86
.LBB0_212:                              ; %.critedge66
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:33792
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_87
	s_branch .LBB0_88
.LBB0_213:                              ; %.critedge68
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:34816
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_90
.LBB0_214:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 34
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a37
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_91
.LBB0_215:                              ; %.critedge70
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:35840
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_92
	s_branch .LBB0_93
.LBB0_216:                              ; %.critedge72
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:36864
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_95
.LBB0_217:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 32
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a39
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_96
.LBB0_218:                              ; %.critedge74
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:37888
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_97
	s_branch .LBB0_98
.LBB0_219:                              ; %.critedge76
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:38912
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_100
.LBB0_220:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 30
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a41
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_101
.LBB0_221:                              ; %.critedge78
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:39936
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_102
	s_branch .LBB0_103
.LBB0_222:                              ; %.critedge80
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:40960
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_105
.LBB0_223:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 28
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a43
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_106
.LBB0_224:                              ; %.critedge82
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:41984
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_107
	s_branch .LBB0_108
.LBB0_225:                              ; %.critedge84
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:43008
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_110
.LBB0_226:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 26
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a45
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_111
.LBB0_227:                              ; %.critedge86
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:44032
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_112
	s_branch .LBB0_113
.LBB0_228:                              ; %.critedge88
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:45056
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_115
.LBB0_229:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 5
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a47
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_116
.LBB0_230:                              ; %.critedge90
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:46080
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_117
	s_branch .LBB0_118
.LBB0_231:                              ; %.critedge92
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:47104
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_120
.LBB0_232:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 7
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a49
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_121
.LBB0_233:                              ; %.critedge94
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:48128
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_122
	s_branch .LBB0_123
.LBB0_234:                              ; %.critedge96
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:49152
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_125
.LBB0_235:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 9
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a51
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_126
.LBB0_236:                              ; %.critedge98
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:50176
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_127
	s_branch .LBB0_128
.LBB0_237:                              ; %.critedge100
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:51200
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_130
.LBB0_238:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 11
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a53
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_131
.LBB0_239:                              ; %.critedge102
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:52224
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_132
	s_branch .LBB0_133
.LBB0_240:                              ; %.critedge104
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:53248
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_135
.LBB0_241:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 13
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a55
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_136
.LBB0_242:                              ; %.critedge106
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:54272
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_137
	s_branch .LBB0_138
.LBB0_243:                              ; %.critedge108
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:55296
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_140
.LBB0_244:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 15
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a57
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_141
.LBB0_245:                              ; %.critedge110
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:56320
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_142
	s_branch .LBB0_143
.LBB0_246:                              ; %.critedge112
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:57344
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_145
.LBB0_247:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 17
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a59
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_146
.LBB0_248:                              ; %.critedge114
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:58368
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_147
	s_branch .LBB0_148
.LBB0_249:                              ; %.critedge116
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:59392
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_150
.LBB0_250:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 19
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a61
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_151
.LBB0_251:                              ; %.critedge118
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:60416
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_152
	s_branch .LBB0_153
.LBB0_252:                              ; %.critedge120
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:61440
	s_andn2_saveexec_b64 s[80:81], s[80:81]
	s_cbranch_execz .LBB0_155
.LBB0_253:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s96, v255, 21
	s_mov_b32 s72, s84
	s_mov_b32 m0, s96
	v_accvgpr_read_b32 v0, a63
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[80:81]
	s_and_saveexec_b64 s[80:81], s[78:79]
	s_xor_b64 s[78:79], exec, s[80:81]
	s_cbranch_execz .LBB0_156
.LBB0_254:                              ; %.critedge122
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:62464
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execnz .LBB0_157
	s_branch .LBB0_158
.LBB0_255:                              ; %.critedge124
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:63488
	s_andn2_saveexec_b64 s[78:79], s[78:79]
	s_cbranch_execz .LBB0_160
.LBB0_256:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s80, v255, 22
	s_mov_b32 s72, s84
	s_mov_b32 m0, s80
	v_accvgpr_read_b32 v0, a65
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_or_b64 exec, exec, s[78:79]
	s_and_saveexec_b64 s[78:79], s[68:69]
	s_xor_b64 s[68:69], exec, s[78:79]
	s_cbranch_execz .LBB0_161
.LBB0_257:                              ; %.critedge126
                                        ;   in Loop: Header=BB0_3 Depth=1
	ds_write_b32 v254, v253 offset:64512
	s_andn2_saveexec_b64 s[68:69], s[68:69]
	s_cbranch_execz .LBB0_2
.LBB0_258:                              ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	v_readlane_b32 s78, v255, 23
	s_mov_b32 s72, s84
	s_mov_b32 m0, s78
	v_accvgpr_read_b32 v0, a66
	buffer_load_dword v0, s[72:75], 0 offen lds
	s_branch .LBB0_2
.LBB0_259:
	.loc	1 0 13 is_stmt 0                ; 00_naive_mfma.py:0:13
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v1, 0
	s_branch .LBB0_261
.LBB0_260:                              ; %._crit_edge.loopexit
	v_accvgpr_read_b32 v0, a152
	v_mov_b64_e32 v[8:9], v[176:177]
	v_accvgpr_read_b32 v1, a153
	v_accvgpr_read_b32 v2, a154
	v_accvgpr_read_b32 v3, a155
	v_mov_b64_e32 v[12:13], v[180:181]
	v_mov_b64_e32 v[10:11], v[178:179]
	.loc	1 99 9 is_stmt 1                ; 00_naive_mfma.py:99:9
	v_cvt_f16_f32_e32 v175, v0
	v_cvt_f16_f32_e32 v122, v1
	v_cvt_f16_f32_e32 v179, v2
	v_mov_b64_e32 v[14:15], v[182:183]
	v_cvt_f16_f32_e32 v181, v3
	v_accvgpr_read_b32 v0, a148
	v_accvgpr_read_b32 v1, a149
	v_accvgpr_read_b32 v2, a150
	v_accvgpr_read_b32 v3, a151
	v_cvt_f16_f32_e32 v86, v0
	v_cvt_f16_f32_e32 v245, v1
	v_cvt_f16_f32_e32 v123, v2
	v_cvt_f16_f32_e32 v182, v3
	v_accvgpr_read_b32 v0, a144
	v_accvgpr_read_b32 v1, a145
	v_accvgpr_read_b32 v2, a146
	v_accvgpr_read_b32 v3, a147
	v_cvt_f16_f32_e32 v244, v0
	v_cvt_f16_f32_e32 v241, v1
	v_cvt_f16_f32_e32 v176, v2
	v_cvt_f16_f32_e32 v183, v3
	v_accvgpr_read_b32 v0, a140
	v_accvgpr_read_b32 v1, a141
	v_accvgpr_read_b32 v2, a142
	v_accvgpr_read_b32 v3, a143
	v_mov_b64_e32 v[16:17], v[184:185]
	v_cvt_f16_f32_e32 v240, v0
	v_cvt_f16_f32_e32 v174, v1
	v_cvt_f16_f32_e32 v88, v2
	v_mov_b64_e32 v[18:19], v[186:187]
	v_cvt_f16_f32_e32 v186, v3
	v_accvgpr_read_b32 v0, a136
	v_accvgpr_read_b32 v1, a137
	v_accvgpr_read_b32 v2, a138
	v_accvgpr_read_b32 v3, a139
	v_cvt_f16_f32_e32 v116, v0
	v_cvt_f16_f32_e32 v84, v1
	v_cvt_f16_f32_e32 v117, v2
	v_cvt_f16_f32_e32 v187, v3
	v_accvgpr_read_b32 v0, a132
	v_accvgpr_read_b32 v1, a133
	v_accvgpr_read_b32 v2, a134
	v_accvgpr_read_b32 v3, a135
	v_mov_b64_e32 v[4:5], v[168:169]
	v_mov_b64_e32 v[20:21], v[188:189]
	v_cvt_f16_f32_e32 v173, v0
	v_mov_b64_e32 v[6:7], v[170:171]
	v_cvt_f16_f32_e32 v170, v1
	v_cvt_f16_f32_e32 v171, v2
	v_mov_b64_e32 v[22:23], v[190:191]
	v_cvt_f16_f32_e32 v190, v3
	v_accvgpr_read_b32 v0, a128
	v_accvgpr_read_b32 v1, a129
	v_accvgpr_read_b32 v2, a130
	v_accvgpr_read_b32 v3, a131
	v_cvt_f16_f32_e32 v168, v0
	v_cvt_f16_f32_e32 v169, v1
	v_cvt_f16_f32_e32 v172, v2
	v_cvt_f16_f32_e32 v191, v3
	v_mov_b64_e32 v[0:1], v[164:165]
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v0, v0
	v_mov_b64_e32 v[32:33], v[196:197]
	v_mov_b64_e32 v[34:35], v[198:199]
	v_accvgpr_write_b32 a36, v20
	v_cvt_f16_f32_e32 v20, v21
	v_accvgpr_write_b32 a25, v0
	v_cvt_f16_f32_e32 v0, v248
	v_accvgpr_read_b32 v26, a124
	v_mov_b64_e32 v[2:3], v[166:167]
	v_cvt_f16_f32_e32 v210, v34
	v_accvgpr_write_b32 a35, v20
	v_cvt_f16_f32_e32 v89, v22
	v_cvt_f16_f32_e32 v34, v23
	v_mov_b64_e32 v[20:21], v[76:77]
	v_accvgpr_read_b32 v27, a125
	v_accvgpr_read_b32 v36, a116
	v_cvt_f16_f32_e32 v12, v12
	v_mov_b64_e32 v[22:23], v[78:79]
	v_cvt_f16_f32_e32 v76, v1
	v_cvt_f16_f32_e32 v77, v2
	v_cvt_f16_f32_e32 v162, v3
	v_accvgpr_write_b32 a24, v0
	v_mov_b64_e32 v[0:1], v[64:65]
	v_cvt_f16_f32_e32 v160, v26
	v_cvt_f16_f32_e32 v165, v27
	v_mov_b64_e32 v[24:25], v[192:193]
	v_accvgpr_read_b32 v38, a118
	v_accvgpr_read_b32 v39, a119
	v_cvt_f16_f32_e32 v0, v0
	v_mov_b64_e32 v[26:27], v[194:195]
	v_cvt_f16_f32_e32 v195, v38
	v_cvt_f16_f32_e32 v197, v39
	v_accvgpr_read_b32 v38, a112
	v_accvgpr_read_b32 v40, a114
	v_accvgpr_read_b32 v41, a115
	v_accvgpr_read_b32 v28, a126
	v_accvgpr_read_b32 v29, a127
	v_cvt_f16_f32_e32 v198, v40
	v_cvt_f16_f32_e32 v199, v41
	v_accvgpr_read_b32 v40, a108
	v_accvgpr_write_b32 a32, v12
	v_cvt_f16_f32_e32 v12, v13
	v_cvt_f16_f32_e32 v254, v148
	v_cvt_f16_f32_e32 v252, v149
	v_cvt_f16_f32_e32 v177, v150
	v_cvt_f16_f32_e32 v178, v151
	v_cvt_f16_f32_e32 v166, v28
	v_cvt_f16_f32_e32 v193, v29
	v_accvgpr_read_b32 v28, a120
	v_accvgpr_read_b32 v39, a113
	v_accvgpr_read_b32 v41, a109
	v_accvgpr_read_b32 v151, a107
	v_accvgpr_write_b32 a23, v0
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v180, v28
	v_cvt_f16_f32_e32 v28, v38
	v_cvt_f16_f32_e32 v209, v39
	v_cvt_f16_f32_e32 v208, v40
	v_cvt_f16_f32_e32 v157, v41
	v_mov_b64_e32 v[38:39], v[200:201]
	v_accvgpr_read_b32 v150, a106
	v_accvgpr_read_b32 v149, a105
	v_accvgpr_read_b32 v148, a104
	v_accvgpr_read_b32 v207, a99
	v_mov_b64_e32 v[40:41], v[202:203]
	v_cvt_f16_f32_e32 v188, v148
	v_cvt_f16_f32_e32 v156, v149
	v_cvt_f16_f32_e32 v159, v150
	v_cvt_f16_f32_e32 v202, v151
	v_accvgpr_read_b32 v151, a103
	v_accvgpr_read_b32 v206, a98
	v_cvt_f16_f32_e32 v203, v151
	v_cvt_f16_f32_e32 v151, v206
	v_cvt_f16_f32_e32 v206, v40
	v_accvgpr_write_b32 a31, v12
	v_cvt_f16_f32_e32 v85, v14
	v_cvt_f16_f32_e32 v40, v15
	v_mov_b64_e32 v[12:13], v[72:73]
	v_accvgpr_write_b32 a22, v0
	v_cvt_f16_f32_e32 v0, v13
	v_mov_b64_e32 v[2:3], v[66:67]
	v_cvt_f16_f32_e32 v189, v39
	v_mov_b64_e32 v[14:15], v[74:75]
	v_cvt_f16_f32_e32 v73, v146
	v_cvt_f16_f32_e32 v146, v136
	v_cvt_f16_f32_e32 v136, v137
	v_cvt_f16_f32_e32 v137, v138
	v_cvt_f16_f32_e32 v138, v139
	v_cvt_f16_f32_e32 v139, v80
	v_cvt_f16_f32_e32 v80, v82
	v_cvt_f16_f32_e32 v39, v2
	v_cvt_f16_f32_e32 v82, v3
	v_accvgpr_write_b32 a21, v0
	v_accvgpr_read_b32 v0, a72
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v8, v8
	v_accvgpr_read_b32 v42, a110
	v_cvt_f16_f32_e32 v158, v42
	v_accvgpr_write_b32 a20, v0
	v_cvt_f16_f32_e32 v0, v96
	v_accvgpr_write_b32 a30, v8
	v_cvt_f16_f32_e32 v8, v9
	v_cvt_f16_f32_e32 v24, v24
	v_accvgpr_write_b32 a19, v0
	v_cvt_f16_f32_e32 v0, v100
	v_accvgpr_write_b32 a29, v8
	v_cvt_f16_f32_e32 v42, v10
	v_cvt_f16_f32_e32 v90, v11
	v_accvgpr_write_b32 a18, v0
	v_cvt_f16_f32_e32 v0, v101
	v_accvgpr_read_b32 v8, a92
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v8, v8
	v_accvgpr_write_b32 a17, v0
	v_cvt_f16_f32_e32 v0, v108
	v_accvgpr_write_b32 a38, v24
	v_cvt_f16_f32_e32 v24, v25
	v_accvgpr_read_b32 v9, a93
	v_accvgpr_write_b32 a16, v0
	v_cvt_f16_f32_e32 v0, v109
	v_accvgpr_write_b32 a34, v16
	v_cvt_f16_f32_e32 v16, v17
	v_accvgpr_write_b32 a28, v8
	v_accvgpr_write_b32 a15, v0
	v_cvt_f16_f32_e32 v0, v112
	v_cvt_f16_f32_e32 v8, v9
	v_cvt_f16_f32_e32 v4, v4
	v_accvgpr_read_b32 v204, a96
	v_accvgpr_write_b32 a14, v0
	v_cvt_f16_f32_e32 v0, v113
	v_cvt_f16_f32_e32 v196, v204
	v_cvt_f16_f32_e32 v204, v32
	v_accvgpr_write_b32 a37, v24
	v_accvgpr_write_b32 a13, v0
	v_cvt_f16_f32_e32 v0, v118
	v_cvt_f16_f32_e32 v91, v26
	v_cvt_f16_f32_e32 v32, v27
	v_accvgpr_read_b32 v10, a94
	v_accvgpr_write_b32 a12, v0
	v_cvt_f16_f32_e32 v0, v119
	v_accvgpr_read_b32 v11, a95
	v_accvgpr_read_b32 v24, a76
	v_accvgpr_read_b32 v29, a121
	v_accvgpr_write_b32 a11, v0
	v_cvt_f16_f32_e32 v0, v152
	v_cvt_f16_f32_e32 v192, v38
	v_accvgpr_write_b32 a33, v16
	v_cvt_f16_f32_e32 v87, v18
	v_cvt_f16_f32_e32 v38, v19
	v_accvgpr_write_b32 a27, v8
	v_cvt_f16_f32_e32 v184, v10
	v_cvt_f16_f32_e32 v185, v11
	v_mov_b64_e32 v[8:9], v[52:53]
	v_mov_b64_e32 v[16:17], v[48:49]
	v_accvgpr_read_b32 v25, a77
	v_accvgpr_read_b32 v26, a78
	v_accvgpr_read_b32 v27, a79
	v_cvt_f16_f32_e32 v164, v29
	v_cvt_f16_f32_e32 v29, v33
	v_accvgpr_write_b32 a26, v4
	v_cvt_f16_f32_e32 v78, v5
	v_cvt_f16_f32_e32 v79, v6
	v_cvt_f16_f32_e32 v161, v7
	v_cvt_f16_f32_e32 v67, v134
	v_mov_b64_e32 v[4:5], v[60:61]
	v_mov_b64_e32 v[10:11], v[54:55]
	v_cvt_f16_f32_e32 v53, v58
	v_mov_b64_e32 v[18:19], v[50:51]
	v_cvt_f16_f32_e32 v134, v24
	v_cvt_f16_f32_e32 v48, v25
	v_cvt_f16_f32_e32 v49, v26
	v_cvt_f16_f32_e32 v58, v27
	v_accvgpr_read_b32 v24, a68
	v_accvgpr_read_b32 v148, a100
	v_accvgpr_read_b32 v205, a97
	v_cvt_f16_f32_e32 v74, v249
	v_cvt_f16_f32_e32 v75, v250
	v_cvt_f16_f32_e32 v163, v251
	v_mov_b64_e32 v[250:251], v[238:239]
	v_mov_b64_e32 v[6:7], v[62:63]
	v_accvgpr_read_b32 v26, a70
	v_cvt_f16_f32_e32 v200, v148
	v_cvt_f16_f32_e32 v148, v205
	v_cvt_f16_f32_e32 v205, v207
	v_cvt_f16_f32_e32 v207, v41
	v_mov_b64_e32 v[248:249], v[236:237]
	v_cvt_f16_f32_e32 v237, v131
	v_cvt_f16_f32_e32 v131, v68
	v_cvt_f16_f32_e32 v68, v71
	v_cvt_f16_f32_e32 v52, v57
	v_cvt_f16_f32_e32 v50, v45
	v_cvt_f16_f32_e32 v57, v47
	v_cvt_f16_f32_e32 v47, v26
	v_cvt_f16_f32_e32 v26, v16
	v_cvt_f16_f32_e32 v45, v18
	v_cvt_f16_f32_e32 v16, v8
	v_cvt_f16_f32_e32 v18, v9
	v_cvt_f16_f32_e32 v41, v6
	v_cvt_f16_f32_e32 v71, v7
	v_accvgpr_read_b32 v6, a80
	v_accvgpr_write_b32 a10, v0
	v_cvt_f16_f32_e32 v0, v153
	v_accvgpr_read_b32 v31, a123
	v_accvgpr_read_b32 v8, a82
	v_cvt_f16_f32_e32 v194, v31
	v_accvgpr_write_b32 a39, v29
	v_cvt_f16_f32_e32 v61, v106
	v_cvt_f16_f32_e32 v31, v8
	v_cvt_f16_f32_e32 v8, v97
	v_cvt_f16_f32_e32 v29, v98
	v_cvt_f16_f32_e32 v106, v99
	v_accvgpr_read_b32 v99, a91
	v_accvgpr_read_b32 v96, a88
	v_accvgpr_write_b32 a9, v0
	v_cvt_f16_f32_e32 v0, v96
	v_accvgpr_read_b32 v97, a89
	v_accvgpr_read_b32 v98, a90
	v_cvt_f16_f32_e32 v236, v135
	v_accvgpr_write_b32 a8, v0
	v_cvt_f16_f32_e32 v0, v97
	v_cvt_f16_f32_e32 v135, v44
	v_cvt_f16_f32_e32 v44, v17
	v_cvt_f16_f32_e32 v17, v98
	v_cvt_f16_f32_e32 v243, v99
	v_accvgpr_read_b32 v99, a87
	v_accvgpr_read_b32 v96, a84
	v_accvgpr_write_b32 a7, v0
	v_cvt_f16_f32_e32 v0, v96
	v_accvgpr_read_b32 v97, a85
	v_accvgpr_read_b32 v30, a122
	v_accvgpr_read_b32 v37, a117
	v_accvgpr_write_b32 a6, v0
	v_cvt_f16_f32_e32 v0, v97
	v_accvgpr_read_b32 v43, a111
	v_accvgpr_read_b32 v150, a102
	v_accvgpr_read_b32 v25, a69
	v_accvgpr_write_b32 a5, v0
	v_cvt_f16_f32_e32 v0, v248
	v_accvgpr_read_b32 v27, a71
	v_accvgpr_read_b32 v1, a73
	v_accvgpr_read_b32 v2, a74
	v_accvgpr_write_b32 a4, v0
	v_cvt_f16_f32_e32 v0, v249
	v_accvgpr_read_b32 v3, a75
	v_accvgpr_read_b32 v7, a81
	v_accvgpr_read_b32 v9, a83
	v_accvgpr_write_b32 a3, v0
	v_cvt_f16_f32_e32 v0, v232
	v_accvgpr_read_b32 v98, a86
	v_cvt_f16_f32_e32 v167, v30
	v_cvt_f16_f32_e32 v30, v37
	v_cvt_f16_f32_e32 v201, v43
	v_cvt_f16_f32_e32 v150, v150
	v_cvt_f16_f32_e32 v211, v35
	v_cvt_f16_f32_e32 v72, v145
	v_cvt_f16_f32_e32 v145, v147
	v_cvt_f16_f32_e32 v147, v141
	v_cvt_f16_f32_e32 v141, v142
	v_cvt_f16_f32_e32 v142, v143
	v_cvt_f16_f32_e32 v66, v133
	v_cvt_f16_f32_e32 v143, v128
	v_cvt_f16_f32_e32 v64, v129
	v_cvt_f16_f32_e32 v65, v130
	v_cvt_f16_f32_e32 v130, v124
	v_cvt_f16_f32_e32 v62, v125
	v_cvt_f16_f32_e32 v63, v126
	v_cvt_f16_f32_e32 v124, v127
	v_cvt_f16_f32_e32 v129, v104
	v_cvt_f16_f32_e32 v60, v105
	v_cvt_f16_f32_e32 v104, v107
	v_cvt_f16_f32_e32 v127, v92
	v_cvt_f16_f32_e32 v126, v93
	v_cvt_f16_f32_e32 v92, v94
	v_cvt_f16_f32_e32 v93, v95
	v_cvt_f16_f32_e32 v133, v81
	v_cvt_f16_f32_e32 v81, v83
	v_cvt_f16_f32_e32 v54, v69
	v_cvt_f16_f32_e32 v55, v70
	v_cvt_f16_f32_e32 v128, v56
	v_cvt_f16_f32_e32 v56, v59
	v_cvt_f16_f32_e32 v51, v46
	v_cvt_f16_f32_e32 v46, v25
	v_cvt_f16_f32_e32 v59, v27
	v_cvt_f16_f32_e32 v69, v19
	v_cvt_f16_f32_e32 v43, v10
	v_cvt_f16_f32_e32 v70, v11
	v_cvt_f16_f32_e32 v10, v5
	v_cvt_f16_f32_e32 v37, v14
	v_cvt_f16_f32_e32 v83, v15
	v_cvt_f16_f32_e32 v14, v20
	v_cvt_f16_f32_e32 v20, v21
	v_cvt_f16_f32_e32 v35, v22
	v_cvt_f16_f32_e32 v94, v23
	v_cvt_f16_f32_e32 v22, v1
	v_cvt_f16_f32_e32 v33, v2
	v_cvt_f16_f32_e32 v95, v3
	v_cvt_f16_f32_e32 v2, v7
	v_cvt_f16_f32_e32 v105, v9
	v_cvt_f16_f32_e32 v27, v102
	v_cvt_f16_f32_e32 v107, v103
	v_cvt_f16_f32_e32 v25, v110
	v_cvt_f16_f32_e32 v125, v111
	v_cvt_f16_f32_e32 v23, v114
	v_cvt_f16_f32_e32 v238, v115
	v_cvt_f16_f32_e32 v21, v120
	v_cvt_f16_f32_e32 v239, v121
	v_cvt_f16_f32_e32 v19, v154
	v_cvt_f16_f32_e32 v242, v155
	v_cvt_f16_f32_e32 v15, v98
	v_cvt_f16_f32_e32 v246, v99
	v_cvt_f16_f32_e32 v13, v250
	v_cvt_f16_f32_e32 v247, v251
	v_accvgpr_write_b32 a2, v0
	v_cvt_f16_f32_e32 v11, v234
	v_cvt_f16_f32_e32 v248, v235
	v_cvt_f16_f32_e32 v9, v230
	v_cvt_f16_f32_e32 v249, v231
	v_cvt_f16_f32_e32 v7, v226
	v_cvt_f16_f32_e32 v250, v227
	v_cvt_f16_f32_e32 v5, v222
	v_cvt_f16_f32_e32 v251, v223
	v_cvt_f16_f32_e32 v3, v218
	v_cvt_f16_f32_e32 v253, v219
	v_cvt_f16_f32_e32 v1, v214
	v_cvt_f16_f32_e32 v0, v215
	v_accvgpr_read_b32 v149, a101
	v_readlane_b32 s12, v255, 1
	v_readlane_b32 s18, v255, 3
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v149, v149
	v_cvt_f16_f32_e32 v144, v144
	v_cvt_f16_f32_e32 v140, v140
	v_cvt_f16_f32_e32 v132, v132
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v6, v6
	v_cvt_f16_f32_e32 v233, v233
	v_cvt_f16_f32_e32 v232, v228
	v_cvt_f16_f32_e32 v229, v229
	v_cvt_f16_f32_e32 v228, v224
	v_cvt_f16_f32_e32 v224, v225
	v_cvt_f16_f32_e32 v220, v220
	v_cvt_f16_f32_e32 v155, v221
	v_cvt_f16_f32_e32 v154, v216
	v_cvt_f16_f32_e32 v153, v217
	v_cvt_f16_f32_e32 v212, v212
	v_cvt_f16_f32_e32 v152, v213
	.loc	1 74 11                         ; 00_naive_mfma.py:74:11
	v_pack_b32_f16 v1, v1, v0
	v_pack_b32_f16 v3, v3, v253
	v_pack_b32_f16 v5, v5, v251
	v_pack_b32_f16 v7, v7, v250
	v_pack_b32_f16 v9, v9, v249
	v_pack_b32_f16 v11, v11, v248
	v_pack_b32_f16 v13, v13, v247
	v_pack_b32_f16 v15, v15, v246
	v_pack_b32_f16 v17, v17, v243
	v_pack_b32_f16 v19, v19, v242
	v_pack_b32_f16 v21, v21, v239
	v_pack_b32_f16 v23, v23, v238
	v_pack_b32_f16 v25, v25, v125
	v_pack_b32_f16 v27, v27, v107
	v_pack_b32_f16 v29, v29, v106
	v_pack_b32_f16 v31, v31, v105
	v_pack_b32_f16 v33, v33, v95
	v_pack_b32_f16 v35, v35, v94
	v_pack_b32_f16 v37, v37, v83
	v_pack_b32_f16 v39, v39, v82
	v_pack_b32_f16 v41, v41, v71
	v_pack_b32_f16 v43, v43, v70
	v_pack_b32_f16 v45, v45, v69
	v_pack_b32_f16 v47, v47, v59
	v_pack_b32_f16 v49, v49, v58
	v_pack_b32_f16 v51, v51, v57
	v_pack_b32_f16 v53, v53, v56
	v_pack_b32_f16 v55, v55, v68
	v_pack_b32_f16 v57, v80, v81
	v_pack_b32_f16 v59, v92, v93
	v_pack_b32_f16 v61, v61, v104
	v_pack_b32_f16 v63, v63, v124
	v_pack_b32_f16 v65, v65, v237
	v_pack_b32_f16 v67, v67, v236
	v_pack_b32_f16 v69, v137, v138
	v_pack_b32_f16 v71, v141, v142
	v_pack_b32_f16 v73, v73, v145
	v_pack_b32_f16 v75, v75, v163
	v_pack_b32_f16 v77, v77, v162
	v_pack_b32_f16 v79, v79, v161
	v_pack_b32_f16 v81, v184, v185
	v_pack_b32_f16 v83, v42, v90
	v_pack_b32_f16 v85, v85, v40
	v_pack_b32_f16 v87, v87, v38
	v_pack_b32_f16 v89, v89, v34
	v_pack_b32_f16 v91, v91, v32
	v_pack_b32_f16 v93, v210, v211
	v_pack_b32_f16 v95, v206, v207
	v_pack_b32_f16 v97, v151, v205
	v_pack_b32_f16 v99, v150, v203
	v_pack_b32_f16 v101, v159, v202
	v_pack_b32_f16 v103, v158, v201
	v_pack_b32_f16 v105, v198, v199
	v_pack_b32_f16 v107, v195, v197
	v_pack_b32_f16 v109, v167, v194
	v_pack_b32_f16 v111, v166, v193
	v_pack_b32_f16 v113, v172, v191
	v_pack_b32_f16 v115, v171, v190
	v_pack_b32_f16 v117, v117, v187
	v_pack_b32_f16 v119, v88, v186
	v_pack_b32_f16 v121, v176, v183
	v_pack_b32_f16 v123, v123, v182
	v_pack_b32_f16 v125, v179, v181
	v_pack_b32_f16 v253, v177, v178
	v_readlane_b32 s13, v255, 2
	v_readlane_b32 s19, v255, 4
	v_accvgpr_read_b32 v104, a0
	v_accvgpr_read_b32 v58, a1
.LBB0_261:                              ; %Flow793
	.loc	1 0 11 is_stmt 0                ; 00_naive_mfma.py:0:11
	s_load_dword s10, s[18:19], 0x2c
	v_readlane_b32 s0, v255, 0
	s_lshr_b32 s0, s0, 6
	.loc	1 103 15 is_stmt 1              ; 00_naive_mfma.py:103:15
	v_and_b32_e32 v0, 15, v104
	s_lshl_b32 s0, s0, 3
	v_and_or_b32 v0, s0, 16, v0
	.loc	1 105 22                        ; 00_naive_mfma.py:105:22
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s33, s10
	.loc	1 105 14 is_stmt 0              ; 00_naive_mfma.py:105:14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	.loc	1 108 15 is_stmt 1              ; 00_naive_mfma.py:108:15
	v_or_b32_e32 v56, s33, v0
	.loc	1 105 14                        ; 00_naive_mfma.py:105:14
	s_add_u32 s2, s12, s0
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[26:27], s14, v56
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_lshrrev_b32_e32 v56, 2, v58
	.loc	1 105 14                        ; 00_naive_mfma.py:105:14
	s_addc_u32 s3, s13, s1
	s_ashr_i32 s77, s76, 31
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_and_b32_e32 v177, 28, v56
	.loc	1 105 14                        ; 00_naive_mfma.py:105:14
	s_lshl_b64 s[0:1], s[76:77], 1
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_or_b32_e32 v56, s76, v177
	.loc	1 105 14                        ; 00_naive_mfma.py:105:14
	s_add_u32 s36, s2, s0
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_cmp_gt_i32_e32 vcc, s15, v56
	.loc	1 105 14                        ; 00_naive_mfma.py:105:14
	s_addc_u32 s0, s3, s1
	.loc	1 106 17                        ; 00_naive_mfma.py:106:17
	v_mul_lo_u32 v56, v0, s10
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	s_mov_b32 s34, 0x5040100
	s_and_b32 s37, s0, 0xffff
	v_add_lshl_u32 v58, v177, v56, 1
	v_bfrev_b32_e32 v178, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[0:1], s[26:27], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v252, v252, v254, s34
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v58, v178, v58, s[0:1]
	v_perm_b32 v124, v122, v175, s34
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v175, 32, v177
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	buffer_store_dwordx2 v[252:253], v58, s[36:39], 0 offen
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_or_b32_e32 v58, s76, v175
	v_cmp_gt_i32_e64 s[0:1], s15, v58
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_add_lshl_u32 v58, v175, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[2:3], s[26:27], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v58, v178, v58, s[2:3]
	buffer_store_dwordx2 v[124:125], v58, s[36:39], 0 offen
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v124, 64, v177
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_or_b32_e32 v58, s76, v124
	v_cmp_gt_i32_e64 s[2:3], s15, v58
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_add_lshl_u32 v58, v124, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[4:5], s[26:27], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v122, v245, v86, s34
	v_cndmask_b32_e64 v58, v178, v58, s[4:5]
	buffer_store_dwordx2 v[122:123], v58, s[36:39], 0 offen
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v122, 0x60, v177
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_or_b32_e32 v70, s76, v122
	v_cmp_gt_i32_e64 s[4:5], s15, v70
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_add_lshl_u32 v70, v122, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[6:7], s[26:27], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v120, v241, v244, s34
	v_cndmask_b32_e64 v70, v178, v70, s[6:7]
	buffer_store_dwordx2 v[120:121], v70, s[36:39], 0 offen
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v120, 0x80, v177
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_or_b32_e32 v92, s76, v120
	v_cmp_gt_i32_e64 s[6:7], s15, v92
	.loc	1 103 15                        ; 00_naive_mfma.py:103:15
	v_or_b32_e32 v58, 32, v0
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_add_lshl_u32 v92, v120, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[8:9], s[26:27], s[6:7]
	.loc	1 103 15                        ; 00_naive_mfma.py:103:15
	v_or_b32_e32 v68, 64, v0
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v118, v174, v240, s34
	v_cndmask_b32_e64 v92, v178, v92, s[8:9]
	.loc	1 106 17                        ; 00_naive_mfma.py:106:17
	v_mul_lo_u32 v94, v58, s10
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_or_b32_e32 v58, s33, v58
	.loc	1 103 15                        ; 00_naive_mfma.py:103:15
	v_or_b32_e32 v70, 0x60, v0
	v_or_b32_e32 v80, 0x80, v0
	v_or_b32_e32 v82, 0xa0, v0
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	buffer_store_dwordx2 v[118:119], v92, s[36:39], 0 offen
	.loc	1 103 15                        ; 00_naive_mfma.py:103:15
	v_or_b32_e32 v92, 0xc0, v0
	v_or_b32_e32 v0, 0xe0, v0
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[30:31], s14, v58
	.loc	1 106 17                        ; 00_naive_mfma.py:106:17
	v_mul_lo_u32 v58, v68, s10
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_or_b32_e32 v68, s33, v68
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v118, 0xa0, v177
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[24:25], s14, v68
	v_or_b32_e32 v68, s33, v70
	.loc	1 106 17                        ; 00_naive_mfma.py:106:17
	v_mul_lo_u32 v182, v0, s10
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_or_b32_e32 v0, s33, v0
	.loc	1 104 15                        ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v119, 0xc0, v177
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[22:23], s14, v68
	v_or_b32_e32 v68, s33, v80
	v_cmp_gt_i32_e64 s[8:9], s14, v0
	.loc	1 108 58 is_stmt 0              ; 00_naive_mfma.py:108:58
	v_or_b32_e32 v0, s76, v118
	.loc	1 104 15 is_stmt 1              ; 00_naive_mfma.py:104:15
	v_or_b32_e32 v121, 0xe0, v177
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[20:21], s14, v68
	v_or_b32_e32 v68, s33, v82
	.loc	1 108 58 is_stmt 0              ; 00_naive_mfma.py:108:58
	v_cmp_gt_i32_e64 s[28:29], s15, v0
	v_or_b32_e32 v0, s76, v119
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[18:19], s14, v68
	v_or_b32_e32 v68, s33, v92
	.loc	1 108 58                        ; 00_naive_mfma.py:108:58
	v_cmp_gt_i32_e64 s[12:13], s15, v0
	v_or_b32_e32 v0, s76, v121
	.loc	1 106 17 is_stmt 1              ; 00_naive_mfma.py:106:17
	v_mul_lo_u32 v123, v70, s10
	v_mul_lo_u32 v125, v80, s10
	v_mul_lo_u32 v179, v82, s10
	v_mul_lo_u32 v181, v92, s10
	.loc	1 108 15                        ; 00_naive_mfma.py:108:15
	v_cmp_gt_i32_e64 s[16:17], s14, v68
	.loc	1 108 58 is_stmt 0              ; 00_naive_mfma.py:108:58
	v_cmp_gt_i32_e64 s[10:11], s15, v0
	.loc	1 110 5 is_stmt 1               ; 00_naive_mfma.py:110:5
	v_add_lshl_u32 v0, v118, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[26:27], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v116, v84, v116, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[116:117], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[26:27], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v114, v170, v173, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[114:115], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v121, v56, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[26:27], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v112, v169, v168, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[112:113], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v94, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v110, v165, v160, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[110:111], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v175, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v108, v164, v180, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[108:109], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v106, v30, v36, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[106:107], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v122, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v104, v209, v28, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[104:105], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v120, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v102, v157, v208, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[102:103], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v118, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v100, v156, v188, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[100:101], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v98, v149, v200, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[98:99], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v121, v94, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[30:31], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v96, v148, v196, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[96:97], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v58, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v94, v189, v192, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[94:95], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a39
	v_perm_b32 v92, v0, v204, s34
	v_add_lshl_u32 v0, v175, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[92:93], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a37
	v_accvgpr_read_b32 v28, a38
	v_perm_b32 v90, v0, v28, s34
	v_add_lshl_u32 v0, v124, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[90:91], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a35
	v_accvgpr_read_b32 v28, a36
	v_perm_b32 v88, v0, v28, s34
	v_add_lshl_u32 v0, v122, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[88:89], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a33
	v_accvgpr_read_b32 v28, a34
	v_perm_b32 v86, v0, v28, s34
	v_add_lshl_u32 v0, v120, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[86:87], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a31
	v_accvgpr_read_b32 v28, a32
	v_perm_b32 v84, v0, v28, s34
	v_add_lshl_u32 v0, v118, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[84:85], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a29
	v_accvgpr_read_b32 v28, a30
	v_perm_b32 v82, v0, v28, s34
	v_add_lshl_u32 v0, v119, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[82:83], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a27
	v_accvgpr_read_b32 v28, a28
	v_perm_b32 v80, v0, v28, s34
	v_add_lshl_u32 v0, v121, v58, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[24:25], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[80:81], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a26
	v_perm_b32 v78, v78, v0, s34
	v_add_lshl_u32 v0, v123, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[78:79], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a25
	v_perm_b32 v76, v76, v0, s34
	v_add_lshl_u32 v0, v175, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[76:77], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a24
	v_perm_b32 v74, v74, v0, s34
	v_add_lshl_u32 v0, v124, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[74:75], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v122, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v72, v72, v144, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[72:73], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v120, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v70, v147, v140, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[70:71], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v118, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v68, v136, v146, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[68:69], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v66, v66, v132, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[66:67], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v121, v123, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[22:23], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v64, v64, v143, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[64:65], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v125, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v62, v62, v130, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[62:63], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v175, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v60, v60, v129, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[60:61], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v58, v126, v127, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[58:59], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v122, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v56, v133, v139, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[56:57], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v120, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v54, v54, v131, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[54:55], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v118, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v52, v52, v128, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[52:53], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v50, v50, v135, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[50:51], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v121, v125, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[20:21], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v48, v48, v134, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[48:49], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v179, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v46, v46, v24, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[46:47], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v175, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v44, v44, v26, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[44:45], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v42, v18, v16, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[42:43], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v122, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v40, v10, v4, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[40:41], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a22
	v_accvgpr_read_b32 v4, a23
	v_perm_b32 v38, v0, v4, s34
	v_add_lshl_u32 v0, v120, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[38:39], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a21
	v_perm_b32 v36, v0, v12, s34
	v_add_lshl_u32 v0, v118, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[36:37], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v34, v20, v14, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[34:35], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a20
	v_perm_b32 v32, v22, v0, s34
	v_add_lshl_u32 v0, v121, v179, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[18:19], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[32:33], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v181, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v30, v2, v6, s34
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[30:31], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a19
	v_perm_b32 v28, v8, v0, s34
	v_add_lshl_u32 v0, v175, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[28:29], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a17
	v_accvgpr_read_b32 v2, a18
	v_perm_b32 v26, v0, v2, s34
	v_add_lshl_u32 v0, v124, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[26:27], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a15
	v_accvgpr_read_b32 v2, a16
	v_perm_b32 v24, v0, v2, s34
	v_add_lshl_u32 v0, v122, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[24:25], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a13
	v_accvgpr_read_b32 v2, a14
	v_perm_b32 v22, v0, v2, s34
	v_add_lshl_u32 v0, v120, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[22:23], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a11
	v_accvgpr_read_b32 v2, a12
	v_perm_b32 v20, v0, v2, s34
	v_add_lshl_u32 v0, v118, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[20:21], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a9
	v_accvgpr_read_b32 v2, a10
	v_perm_b32 v18, v0, v2, s34
	v_add_lshl_u32 v0, v119, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[18:19], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a7
	v_accvgpr_read_b32 v2, a8
	v_perm_b32 v16, v0, v2, s34
	v_add_lshl_u32 v0, v121, v181, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 s[14:15], s[16:17], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e64 v0, v178, v0, s[14:15]
	buffer_store_dwordx2 v[16:17], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a5
	v_accvgpr_read_b32 v2, a6
	v_perm_b32 v14, v0, v2, s34
	v_add_lshl_u32 v0, v182, v177, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], vcc
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[14:15], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a3
	v_accvgpr_read_b32 v2, a4
	v_perm_b32 v12, v0, v2, s34
	v_add_lshl_u32 v0, v175, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[0:1]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[12:13], v0, s[36:39], 0 offen
	v_accvgpr_read_b32 v0, a2
	v_perm_b32 v10, v233, v0, s34
	v_add_lshl_u32 v0, v124, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[2:3]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[10:11], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v122, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[4:5]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v8, v229, v232, s34
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[8:9], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v120, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[6:7]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v6, v224, v228, s34
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[6:7], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v118, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[28:29]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v4, v155, v220, s34
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[4:5], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[12:13]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v2, v153, v154, s34
	v_cndmask_b32_e32 v0, v178, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[36:39], 0 offen
	v_add_lshl_u32 v2, v121, v182, 1
	.loc	1 108 14                        ; 00_naive_mfma.py:108:14
	s_and_b64 vcc, s[8:9], s[10:11]
	.loc	1 110 5                         ; 00_naive_mfma.py:110:5
	v_perm_b32 v0, v152, v212, s34
	v_cndmask_b32_e32 v2, v178, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[36:39], 0 offen
	.loc	1 8 1                           ; 00_naive_mfma.py:8:1
	s_endpgm
.Ltmp11:
.Lfunc_end0:
	.size	v0_naive, .Lfunc_end0-v0_naive
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v0_naive
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 8
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 416
		.amdhsa_next_free_sgpr 100
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
	.set .Lv0_naive.num_agpr, 160
	.set .Lv0_naive.numbered_sgpr, 100
	.set .Lv0_naive.num_named_barrier, 0
	.set .Lv0_naive.private_seg_size, 0
	.set .Lv0_naive.uses_vcc, 1
	.set .Lv0_naive.uses_flat_scratch, 0
	.set .Lv0_naive.has_dyn_sized_stack, 0
	.set .Lv0_naive.has_recursion, 0
	.set .Lv0_naive.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 19932
; TotalNumSgprs: 106
; NumVgprs: 256
; NumAgprs: 160
; TotalNumVgprs: 416
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 51
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 416
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
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
	.byte	28                              ; DW_AT_call_line
	.byte	17                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	76                              ; DW_AT_call_line
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
  - .agpr_count:     160
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
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         144
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         152
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         160
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         168
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         176
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         184
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         264
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .max_flat_workgroup_size: 256
    .name:           v0_naive
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 56
    .symbol:         v0_naive.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     416
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
