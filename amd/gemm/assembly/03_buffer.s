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
; %bb.6:
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
; %bb.7:
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
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s19, 0, s1
	s_abs_i32 s18, s16
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_readfirstlane_b32 s15, v0
	.loc	1 29 13                         ; 00_naive_mfma.py:29:13
	v_rcp_f32_e32 v1, v1
	s_xor_b32 s14, s16, s0
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	s_and_b32 s17, s15, 0xc0
	.loc	1 29 13                         ; 00_naive_mfma.py:29:13
	s_ashr_i32 s14, s14, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_and_or_b32 v18, v0, 63, s17
	.loc	1 29 13                         ; 00_naive_mfma.py:29:13
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s18, s20
	s_mul_i32 s20, s19, s1
	s_sub_i32 s18, s18, s20
	s_add_i32 s20, s19, 1
	s_sub_i32 s21, s18, s1
	s_cmp_ge_u32 s18, s1
	s_cselect_b32 s19, s20, s19
	s_cselect_b32 s18, s21, s18
	s_add_i32 s20, s19, 1
	s_cmp_ge_u32 s18, s1
	s_cselect_b32 s1, s20, s19
	s_xor_b32 s1, s1, s14
	s_sub_i32 s1, s1, s14
	.loc	1 30 13                         ; 00_naive_mfma.py:30:13
	s_mul_i32 s0, s1, s0
	s_sub_i32 s0, s16, s0
	.loc	1 52 22                         ; 00_naive_mfma.py:52:22
	s_lshl_b32 s16, s1, 8
	.loc	1 53 22                         ; 00_naive_mfma.py:53:22
	s_lshl_b32 s14, s0, 8
.Ltmp3:
	.loc	2 43 13                         ; standard.py:43:13 @[ 00_naive_mfma.py:70:23 ]
	s_add_i32 s0, s10, 63
.Ltmp4:
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_cmp_lt_i32 s0, 64
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %.lr.ph
	.loc	1 47 15                         ; 00_naive_mfma.py:47:15
	v_and_b32_e32 v17, 7, v0
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_lshrrev_b32_e32 v1, 3, v18
	.loc	1 47 15                         ; 00_naive_mfma.py:47:15
	v_lshlrev_b32_e32 v19, 3, v17
	.loc	1 46 15                         ; 00_naive_mfma.py:46:15
	v_or_b32_e32 v2, 0xe0, v1
	v_or_b32_e32 v4, 0xc0, v1
	v_or_b32_e32 v6, 0xa0, v1
	v_or_b32_e32 v8, 0x80, v1
	v_or_b32_e32 v10, 0x60, v1
	v_or_b32_e32 v12, 64, v1
	v_or_b32_e32 v14, 32, v1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v16, v1, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v1, v1, s11
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v15, v14, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v14, v14, s11
	v_add_lshl_u32 v1, v1, v19, 1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v13, v12, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v12, v12, s11
	v_accvgpr_write_b32 a111, v1
	v_add_lshl_u32 v1, v14, v19, 1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v11, v10, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v10, v10, s11
	v_accvgpr_write_b32 a112, v1
	v_add_lshl_u32 v1, v12, v19, 1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v9, v8, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v8, v8, s11
	v_accvgpr_write_b32 a113, v1
	v_add_lshl_u32 v1, v10, v19, 1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v7, v6, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v6, v6, s11
	v_accvgpr_write_b32 a114, v1
	v_add_lshl_u32 v1, v8, v19, 1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v5, v4, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v4, v4, s11
	v_accvgpr_write_b32 a115, v1
	v_add_lshl_u32 v1, v6, v19, 1
	.loc	1 56 48                         ; 00_naive_mfma.py:56:48
	v_mul_lo_u32 v3, v2, s12
	.loc	1 55 17                         ; 00_naive_mfma.py:55:17
	v_mul_lo_u32 v2, v2, s11
	v_accvgpr_write_b32 a116, v1
	v_add_lshl_u32 v1, v4, v19, 1
	v_accvgpr_write_b32 a117, v1
	v_add_lshl_u32 v1, v2, v19, 1
.Ltmp5:
	.loc	2 43 12                         ; standard.py:43:12 @[ 00_naive_mfma.py:70:23 ]
	s_ashr_i32 s1, s0, 31
	v_accvgpr_write_b32 a118, v1
	v_add_lshl_u32 v1, v16, v19, 1
	s_lshr_b32 s1, s1, 26
	v_accvgpr_write_b32 a119, v1
	v_add_lshl_u32 v1, v15, v19, 1
	s_add_i32 s0, s0, s1
	v_accvgpr_write_b32 a120, v1
	v_add_lshl_u32 v1, v13, v19, 1
	s_ashr_i32 s17, s0, 6
.Ltmp6:
	.loc	1 53 22                         ; 00_naive_mfma.py:53:22
	s_mul_i32 s0, s14, s12
	v_accvgpr_write_b32 a121, v1
	v_add_lshl_u32 v1, v11, v19, 1
	.loc	1 53 14 is_stmt 0               ; 00_naive_mfma.py:53:14
	s_ashr_i32 s1, s0, 31
	v_accvgpr_write_b32 a122, v1
	v_add_lshl_u32 v1, v9, v19, 1
	s_lshl_b64 s[0:1], s[0:1], 1
	v_accvgpr_write_b32 a123, v1
	v_add_lshl_u32 v1, v7, v19, 1
	s_add_u32 s4, s4, s0
	.loc	1 52 22 is_stmt 1               ; 00_naive_mfma.py:52:22
	s_mul_i32 s0, s16, s11
	v_accvgpr_write_b32 a124, v1
	v_add_lshl_u32 v1, v5, v19, 1
	.loc	1 53 14                         ; 00_naive_mfma.py:53:14
	s_addc_u32 s5, s5, s1
	.loc	1 52 14                         ; 00_naive_mfma.py:52:14
	s_ashr_i32 s1, s0, 31
	v_accvgpr_write_b32 a125, v1
	v_add_lshl_u32 v1, v3, v19, 1
	s_lshl_b64 s[0:1], s[0:1], 1
	v_accvgpr_write_b32 a126, v1
	v_and_b32_e32 v1, 24, v0
	s_add_u32 s11, s2, s0
	v_lshlrev_b32_e32 v2, 2, v18
	v_lshlrev_b32_e32 v3, 1, v1
	s_movk_i32 s0, 0x380
	s_addc_u32 s12, s3, s1
	v_and_or_b32 v3, v2, s0, v3
	v_lshlrev_b32_e32 v5, 5, v0
	v_bfe_i32 v8, v0, 4, 1
	s_lshl_b32 s0, s15, 2
	v_and_b32_e32 v4, 3, v0
	v_and_b32_e32 v6, 0x180, v5
	v_and_b32_e32 v9, 0x1040, v8
	s_and_b32 s0, s0, 0x200
	v_lshlrev_b32_e32 v12, 4, v17
	v_lshlrev_b32_e32 v7, 4, v4
	v_lshl_or_b32 v4, v4, 10, v6
	v_or_b32_e32 v6, s0, v9
	v_lshlrev_b32_e32 v11, 10, v0
	s_movk_i32 s0, 0x1800
	v_lshl_or_b32 v1, v1, 7, v12
	v_lshrrev_b32_e32 v10, 1, v0
	v_accvgpr_write_b32 a108, v0
	v_and_b32_e32 v2, 0x3e0, v2
	v_and_or_b32 v0, v11, s0, v12
	s_movk_i32 s0, 0x410
	v_and_b32_e32 v5, 0x1e0, v5
	v_xor_b32_e32 v1, v3, v1
	v_and_b32_e32 v10, 16, v10
	v_xor_b32_e32 v0, v2, v0
	s_lshl_b32 s1, s15, 3
	v_and_or_b32 v5, v8, s0, v5
	v_xor_b32_e32 v3, 0x1040, v1
	v_xor_b32_e32 v7, v7, v10
	v_xor_b32_e32 v2, 0x410, v0
	s_and_b32 s1, s1, 0x200
	v_xor_b32_e32 v5, v5, v10
	v_add_u32_e32 v1, 0, v1
	v_add_u32_e32 v0, 0, v0
	v_or3_b32 v4, v6, v4, v7
	v_or_b32_e32 v5, s1, v5
	v_accvgpr_write_b32 a127, v1
	v_add_u32_e32 v1, 0, v3
	v_accvgpr_write_b32 a133, v0
	v_add_u32_e32 v0, 0, v2
	v_xor_b32_e32 v6, 32, v4
	v_xor_b32_e32 v8, 0x820, v5
	v_accvgpr_write_b32 a128, v1
	v_add_u32_e32 v1, 0, v4
	v_accvgpr_write_b32 a134, v0
	v_add_u32_e32 v0, 0, v5
	v_xor_b32_e32 v7, 64, v4
	v_xor_b32_e32 v10, 0x1040, v5
	v_accvgpr_write_b32 a129, v1
	v_add_u32_e32 v1, 0, v6
	v_accvgpr_write_b32 a135, v0
	v_add_u32_e32 v0, 0, v8
	v_xor_b32_e32 v9, 0x60, v4
	v_xor_b32_e32 v11, 0x1860, v5
	v_mov_b32_e32 v78, 0
	v_accvgpr_write_b32 a130, v1
	v_add_u32_e32 v1, 0, v7
	v_accvgpr_write_b32 a136, v0
	v_add_u32_e32 v0, 0, v10
	v_accvgpr_write_b32 a131, v1
	v_add_u32_e32 v1, 0, v9
	v_accvgpr_write_b32 a138, v0
	v_add_u32_e32 v0, 0, v11
	v_mov_b32_e32 v3, v78
	v_accvgpr_write_b32 a132, v1
	v_accvgpr_write_b32 a137, v0
	v_mov_b32_e32 v0, v78
	v_mov_b32_e32 v1, v78
	v_mov_b32_e32 v2, v78
	v_accvgpr_write_b32 a143, v3
	v_accvgpr_write_b32 a147, v3
	v_accvgpr_write_b32 a151, v3
	v_accvgpr_write_b32 a155, v3
	v_accvgpr_write_b32 a159, v3
	v_accvgpr_write_b32 a163, v3
	v_accvgpr_write_b32 a167, v3
	v_accvgpr_write_b32 a171, v3
	v_accvgpr_write_b32 a175, v3
	v_accvgpr_write_b32 a179, v3
	v_accvgpr_write_b32 a183, v3
	v_accvgpr_write_b32 a187, v3
	v_accvgpr_write_b32 a191, v3
	v_accvgpr_write_b32 a195, v3
	v_accvgpr_write_b32 a110, v19
	v_accvgpr_write_b32 a109, v18
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_mov_b32_e32 v79, v78
	v_mov_b32_e32 v80, v78
	v_mov_b32_e32 v81, v78
	v_accvgpr_write_b32 a142, v2
	v_accvgpr_write_b32 a141, v1
	v_accvgpr_write_b32 a140, v0
	v_accvgpr_write_b32 a146, v2
	v_accvgpr_write_b32 a145, v1
	v_accvgpr_write_b32 a144, v0
	v_accvgpr_write_b32 a150, v2
	v_accvgpr_write_b32 a149, v1
	v_accvgpr_write_b32 a148, v0
	v_mov_b32_e32 v16, v78
	v_mov_b32_e32 v17, v78
	v_mov_b32_e32 v18, v78
	v_mov_b32_e32 v19, v78
	v_mov_b32_e32 v12, v78
	v_mov_b32_e32 v13, v78
	v_mov_b32_e32 v14, v78
	v_mov_b32_e32 v15, v78
	v_mov_b32_e32 v20, v78
	v_mov_b32_e32 v21, v78
	v_mov_b32_e32 v22, v78
	v_mov_b32_e32 v23, v78
	v_mov_b32_e32 v24, v78
	v_mov_b32_e32 v25, v78
	v_mov_b32_e32 v26, v78
	v_mov_b32_e32 v27, v78
	v_mov_b32_e32 v28, v78
	v_mov_b32_e32 v29, v78
	v_mov_b32_e32 v30, v78
	v_mov_b32_e32 v31, v78
	v_mov_b32_e32 v32, v78
	v_mov_b32_e32 v33, v78
	v_mov_b32_e32 v34, v78
	v_mov_b32_e32 v35, v78
	v_mov_b32_e32 v36, v78
	v_mov_b32_e32 v37, v78
	v_mov_b32_e32 v38, v78
	v_mov_b32_e32 v39, v78
	v_accvgpr_write_b32 a154, v2
	v_accvgpr_write_b32 a153, v1
	v_accvgpr_write_b32 a152, v0
	v_accvgpr_write_b32 a158, v2
	v_accvgpr_write_b32 a157, v1
	v_accvgpr_write_b32 a156, v0
	v_accvgpr_write_b32 a162, v2
	v_accvgpr_write_b32 a161, v1
	v_accvgpr_write_b32 a160, v0
	v_accvgpr_write_b32 a166, v2
	v_accvgpr_write_b32 a165, v1
	v_accvgpr_write_b32 a164, v0
	v_accvgpr_write_b32 a170, v2
	v_accvgpr_write_b32 a169, v1
	v_accvgpr_write_b32 a168, v0
	v_accvgpr_write_b32 a174, v2
	v_accvgpr_write_b32 a173, v1
	v_accvgpr_write_b32 a172, v0
	v_accvgpr_write_b32 a178, v2
	v_accvgpr_write_b32 a177, v1
	v_accvgpr_write_b32 a176, v0
	v_accvgpr_write_b32 a182, v2
	v_accvgpr_write_b32 a181, v1
	v_accvgpr_write_b32 a180, v0
	v_accvgpr_write_b32 a186, v2
	v_accvgpr_write_b32 a185, v1
	v_accvgpr_write_b32 a184, v0
	v_accvgpr_write_b32 a190, v2
	v_accvgpr_write_b32 a189, v1
	v_accvgpr_write_b32 a188, v0
	v_mov_b32_e32 v86, v78
	v_mov_b32_e32 v87, v78
	v_mov_b32_e32 v88, v78
	v_mov_b32_e32 v89, v78
	v_mov_b32_e32 v90, v78
	v_mov_b32_e32 v91, v78
	v_mov_b32_e32 v92, v78
	v_mov_b32_e32 v93, v78
	v_mov_b32_e32 v94, v78
	v_mov_b32_e32 v95, v78
	v_mov_b32_e32 v96, v78
	v_mov_b32_e32 v97, v78
	v_mov_b32_e32 v98, v78
	v_mov_b32_e32 v99, v78
	v_mov_b32_e32 v100, v78
	v_mov_b32_e32 v101, v78
	v_mov_b32_e32 v102, v78
	v_mov_b32_e32 v103, v78
	v_mov_b32_e32 v104, v78
	v_mov_b32_e32 v105, v78
	v_mov_b32_e32 v106, v78
	v_mov_b32_e32 v107, v78
	v_mov_b32_e32 v108, v78
	v_mov_b32_e32 v109, v78
	v_mov_b32_e32 v110, v78
	v_mov_b32_e32 v111, v78
	v_mov_b32_e32 v112, v78
	v_mov_b32_e32 v113, v78
	v_mov_b32_e32 v114, v78
	v_mov_b32_e32 v115, v78
	v_mov_b32_e32 v116, v78
	v_mov_b32_e32 v117, v78
	v_mov_b32_e32 v118, v78
	v_mov_b32_e32 v119, v78
	v_mov_b32_e32 v120, v78
	v_mov_b32_e32 v121, v78
	v_mov_b32_e32 v122, v78
	v_mov_b32_e32 v123, v78
	v_mov_b32_e32 v124, v78
	v_mov_b32_e32 v125, v78
	v_mov_b32_e32 v126, v78
	v_mov_b32_e32 v127, v78
	v_mov_b32_e32 v128, v78
	v_mov_b32_e32 v129, v78
	v_mov_b32_e32 v130, v78
	v_mov_b32_e32 v131, v78
	v_mov_b32_e32 v132, v78
	v_mov_b32_e32 v133, v78
	v_mov_b32_e32 v134, v78
	v_mov_b32_e32 v135, v78
	v_mov_b32_e32 v136, v78
	v_mov_b32_e32 v137, v78
	v_mov_b32_e32 v138, v78
	v_mov_b32_e32 v139, v78
	v_mov_b32_e32 v140, v78
	v_mov_b32_e32 v141, v78
	v_mov_b32_e32 v142, v78
	v_mov_b32_e32 v143, v78
	v_mov_b32_e32 v144, v78
	v_mov_b32_e32 v145, v78
	v_mov_b32_e32 v146, v78
	v_mov_b32_e32 v147, v78
	v_mov_b32_e32 v148, v78
	v_mov_b32_e32 v149, v78
	v_mov_b32_e32 v150, v78
	v_mov_b32_e32 v151, v78
	v_mov_b32_e32 v152, v78
	v_mov_b32_e32 v153, v78
	v_mov_b32_e32 v154, v78
	v_mov_b32_e32 v155, v78
	v_mov_b32_e32 v156, v78
	v_mov_b32_e32 v157, v78
	v_mov_b32_e32 v158, v78
	v_mov_b32_e32 v159, v78
	v_mov_b32_e32 v160, v78
	v_mov_b32_e32 v161, v78
	v_mov_b32_e32 v162, v78
	v_mov_b32_e32 v163, v78
	v_mov_b32_e32 v164, v78
	v_mov_b32_e32 v165, v78
	v_mov_b32_e32 v166, v78
	v_mov_b32_e32 v167, v78
	v_mov_b32_e32 v168, v78
	v_mov_b32_e32 v169, v78
	v_mov_b32_e32 v170, v78
	v_mov_b32_e32 v171, v78
	v_mov_b32_e32 v172, v78
	v_mov_b32_e32 v173, v78
	v_mov_b32_e32 v174, v78
	v_mov_b32_e32 v175, v78
	v_mov_b32_e32 v176, v78
	v_mov_b32_e32 v177, v78
	v_mov_b32_e32 v178, v78
	v_mov_b32_e32 v179, v78
	v_mov_b32_e32 v180, v78
	v_mov_b32_e32 v181, v78
	v_mov_b32_e32 v182, v78
	v_mov_b32_e32 v183, v78
	v_mov_b32_e32 v184, v78
	v_mov_b32_e32 v185, v78
	v_mov_b32_e32 v186, v78
	v_mov_b32_e32 v187, v78
	v_mov_b32_e32 v188, v78
	v_mov_b32_e32 v189, v78
	v_mov_b32_e32 v190, v78
	v_mov_b32_e32 v191, v78
	v_mov_b32_e32 v192, v78
	v_mov_b32_e32 v193, v78
	v_mov_b32_e32 v194, v78
	v_mov_b32_e32 v195, v78
	v_mov_b32_e32 v196, v78
	v_mov_b32_e32 v197, v78
	v_mov_b32_e32 v198, v78
	v_mov_b32_e32 v199, v78
	v_mov_b32_e32 v200, v78
	v_mov_b32_e32 v201, v78
	v_mov_b32_e32 v202, v78
	v_mov_b32_e32 v203, v78
	v_mov_b32_e32 v204, v78
	v_mov_b32_e32 v205, v78
	v_mov_b32_e32 v206, v78
	v_mov_b32_e32 v207, v78
	v_mov_b32_e32 v208, v78
	v_mov_b32_e32 v209, v78
	v_mov_b32_e32 v210, v78
	v_mov_b32_e32 v211, v78
	v_mov_b32_e32 v212, v78
	v_mov_b32_e32 v213, v78
	v_mov_b32_e32 v214, v78
	v_mov_b32_e32 v215, v78
	v_mov_b32_e32 v216, v78
	v_mov_b32_e32 v217, v78
	v_mov_b32_e32 v218, v78
	v_mov_b32_e32 v219, v78
	v_mov_b32_e32 v220, v78
	v_mov_b32_e32 v221, v78
	v_mov_b32_e32 v222, v78
	v_mov_b32_e32 v223, v78
	v_mov_b32_e32 v224, v78
	v_mov_b32_e32 v225, v78
	v_mov_b32_e32 v226, v78
	v_mov_b32_e32 v227, v78
	v_mov_b32_e32 v228, v78
	v_mov_b32_e32 v229, v78
	v_mov_b32_e32 v230, v78
	v_mov_b32_e32 v231, v78
	v_mov_b32_e32 v232, v78
	v_mov_b32_e32 v233, v78
	v_mov_b32_e32 v234, v78
	v_mov_b32_e32 v235, v78
	v_mov_b32_e32 v236, v78
	v_mov_b32_e32 v237, v78
	v_mov_b32_e32 v238, v78
	v_mov_b32_e32 v239, v78
	v_mov_b32_e32 v240, v78
	v_mov_b32_e32 v241, v78
	v_mov_b32_e32 v242, v78
	v_mov_b32_e32 v243, v78
	v_mov_b32_e32 v244, v78
	v_mov_b32_e32 v245, v78
	v_mov_b32_e32 v246, v78
	v_mov_b32_e32 v247, v78
	v_mov_b32_e32 v248, v78
	v_mov_b32_e32 v249, v78
	v_mov_b32_e32 v250, v78
	v_mov_b32_e32 v251, v78
	v_mov_b32_e32 v252, v78
	v_mov_b32_e32 v253, v78
	v_accvgpr_write_b32 a194, v2
	v_accvgpr_write_b32 a193, v1
	v_accvgpr_write_b32 a192, v0
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 0 14 is_stmt 0                ; 00_naive_mfma.py:0:14
	v_accvgpr_read_b32 v0, a110
	.loc	1 75 49 is_stmt 1               ; 00_naive_mfma.py:75:49
	v_cmp_gt_i32_e32 vcc, s10, v0
	v_bfrev_b32_e32 v11, 1
	v_accvgpr_read_b32 v10, a121
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	v_cndmask_b32_e32 v50, v11, v10, vcc
	v_accvgpr_read_b32 v10, a122
	v_cndmask_b32_e32 v54, v11, v10, vcc
	v_accvgpr_read_b32 v10, a123
	v_cndmask_b32_e32 v70, v11, v10, vcc
	v_accvgpr_read_b32 v10, a124
	v_accvgpr_read_b32 v0, a111
	v_cndmask_b32_e32 v72, v11, v10, vcc
	v_accvgpr_read_b32 v10, a125
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	s_and_b32 s1, s12, 0xffff
	s_mov_b32 s0, s11
	v_cndmask_b32_e32 v0, v11, v0, vcc
	v_accvgpr_read_b32 v1, a112
	v_accvgpr_read_b32 v2, a113
	v_accvgpr_read_b32 v3, a114
	v_accvgpr_read_b32 v4, a115
	v_accvgpr_read_b32 v5, a116
	v_accvgpr_read_b32 v6, a117
	v_accvgpr_read_b32 v7, a118
	v_accvgpr_read_b32 v8, a119
	v_accvgpr_read_b32 v9, a120
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	v_cndmask_b32_e32 v74, v11, v10, vcc
	v_accvgpr_read_b32 v10, a126
	v_mov_b64_e32 v[84:85], v[14:15]
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	v_cndmask_b32_e32 v1, v11, v1, vcc
	v_cndmask_b32_e32 v2, v11, v2, vcc
	v_cndmask_b32_e32 v3, v11, v3, vcc
	v_cndmask_b32_e32 v4, v11, v4, vcc
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e32 v6, v11, v6, vcc
	v_cndmask_b32_e32 v7, v11, v7, vcc
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	v_cndmask_b32_e32 v8, v11, v8, vcc
	v_cndmask_b32_e32 v9, v11, v9, vcc
	v_cndmask_b32_e32 v75, v11, v10, vcc
	v_mov_b64_e32 v[82:83], v[12:13]
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	buffer_load_dwordx4 v[10:13], v0, s[0:3], 0 offen
	v_accvgpr_write_b32 a196, v230
	v_accvgpr_write_b32 a211, v209
	v_accvgpr_write_b32 a200, v234
	v_accvgpr_write_b32 a197, v231
	v_accvgpr_write_b32 a198, v232
	v_accvgpr_write_b32 a199, v233
	v_accvgpr_write_b32 a210, v208
	v_accvgpr_write_b32 a209, v207
	v_accvgpr_write_b32 a208, v206
	v_accvgpr_write_b32 a201, v235
	v_accvgpr_write_b32 a202, v236
	v_accvgpr_write_b32 a203, v237
	v_accvgpr_write_b32 a204, v238
	v_mov_b64_e32 v[234:235], v[242:243]
	v_mov_b64_e32 v[230:231], v[246:247]
	v_mov_b64_e32 v[206:207], v[250:251]
	v_accvgpr_write_b32 a215, v213
	v_accvgpr_write_b32 a205, v239
	v_accvgpr_write_b32 a206, v240
	v_accvgpr_write_b32 a207, v241
	v_mov_b64_e32 v[240:241], v[216:217]
	v_mov_b64_e32 v[236:237], v[244:245]
	v_mov_b64_e32 v[244:245], v[220:221]
	v_mov_b64_e32 v[232:233], v[248:249]
	v_mov_b64_e32 v[248:249], v[224:225]
	v_mov_b64_e32 v[208:209], v[252:253]
	v_mov_b64_e32 v[252:253], v[228:229]
	v_accvgpr_write_b32 a219, v185
	v_accvgpr_write_b32 a214, v212
	v_accvgpr_write_b32 a213, v211
	v_accvgpr_write_b32 a212, v210
	v_mov_b64_e32 v[212:213], v[188:189]
	v_mov_b64_e32 v[238:239], v[214:215]
	v_mov_b64_e32 v[216:217], v[192:193]
	v_mov_b64_e32 v[242:243], v[218:219]
	v_mov_b64_e32 v[220:221], v[196:197]
	v_mov_b64_e32 v[246:247], v[222:223]
	v_mov_b64_e32 v[224:225], v[200:201]
	v_mov_b64_e32 v[250:251], v[226:227]
	v_mov_b64_e32 v[228:229], v[204:205]
	v_accvgpr_write_b32 a218, v184
	v_accvgpr_write_b32 a217, v183
	v_accvgpr_write_b32 a216, v182
	v_mov_b64_e32 v[184:185], v[160:161]
	v_mov_b64_e32 v[210:211], v[186:187]
	v_mov_b64_e32 v[188:189], v[164:165]
	v_mov_b64_e32 v[214:215], v[190:191]
	v_mov_b64_e32 v[192:193], v[168:169]
	v_mov_b64_e32 v[218:219], v[194:195]
	v_mov_b64_e32 v[196:197], v[172:173]
	v_mov_b64_e32 v[222:223], v[198:199]
	v_mov_b64_e32 v[200:201], v[176:177]
	v_mov_b64_e32 v[226:227], v[202:203]
	v_mov_b64_e32 v[204:205], v[180:181]
	v_mov_b64_e32 v[182:183], v[158:159]
	v_mov_b64_e32 v[160:161], v[136:137]
	v_mov_b64_e32 v[186:187], v[162:163]
	v_mov_b64_e32 v[164:165], v[140:141]
	v_mov_b64_e32 v[190:191], v[166:167]
	v_mov_b64_e32 v[168:169], v[144:145]
	v_mov_b64_e32 v[194:195], v[170:171]
	v_mov_b64_e32 v[172:173], v[148:149]
	v_mov_b64_e32 v[198:199], v[174:175]
	v_mov_b64_e32 v[176:177], v[152:153]
	v_mov_b64_e32 v[202:203], v[178:179]
	v_mov_b64_e32 v[180:181], v[156:157]
	v_mov_b64_e32 v[158:159], v[134:135]
	v_mov_b64_e32 v[136:137], v[112:113]
	v_mov_b64_e32 v[162:163], v[138:139]
	v_mov_b64_e32 v[140:141], v[116:117]
	v_mov_b64_e32 v[166:167], v[142:143]
	v_mov_b64_e32 v[144:145], v[120:121]
	v_mov_b64_e32 v[170:171], v[146:147]
	v_mov_b64_e32 v[148:149], v[124:125]
	v_mov_b64_e32 v[174:175], v[150:151]
	v_mov_b64_e32 v[152:153], v[128:129]
	v_mov_b64_e32 v[178:179], v[154:155]
	v_mov_b64_e32 v[156:157], v[132:133]
	v_mov_b64_e32 v[134:135], v[110:111]
	v_mov_b64_e32 v[112:113], v[88:89]
	v_mov_b64_e32 v[138:139], v[114:115]
	v_mov_b64_e32 v[116:117], v[92:93]
	v_mov_b64_e32 v[142:143], v[118:119]
	v_mov_b64_e32 v[120:121], v[96:97]
	v_mov_b64_e32 v[146:147], v[122:123]
	v_mov_b64_e32 v[124:125], v[100:101]
	v_mov_b64_e32 v[150:151], v[126:127]
	v_mov_b64_e32 v[128:129], v[104:105]
	v_mov_b64_e32 v[154:155], v[130:131]
	v_mov_b64_e32 v[132:133], v[108:109]
	v_mov_b64_e32 v[110:111], v[86:87]
	v_mov_b64_e32 v[88:89], v[18:19]
	v_mov_b64_e32 v[114:115], v[90:91]
	v_mov_b64_e32 v[92:93], v[22:23]
	v_mov_b64_e32 v[118:119], v[94:95]
	v_mov_b64_e32 v[96:97], v[26:27]
	v_mov_b64_e32 v[122:123], v[98:99]
	v_mov_b64_e32 v[100:101], v[30:31]
	v_mov_b64_e32 v[126:127], v[102:103]
	v_mov_b64_e32 v[104:105], v[34:35]
	v_mov_b64_e32 v[130:131], v[106:107]
	v_mov_b64_e32 v[108:109], v[38:39]
	v_mov_b64_e32 v[86:87], v[16:17]
	buffer_load_dwordx4 v[14:17], v1, s[0:3], 0 offen
	v_mov_b64_e32 v[90:91], v[20:21]
	buffer_load_dwordx4 v[18:21], v2, s[0:3], 0 offen
	v_mov_b64_e32 v[94:95], v[24:25]
	buffer_load_dwordx4 v[22:25], v3, s[0:3], 0 offen
	v_mov_b64_e32 v[98:99], v[28:29]
	buffer_load_dwordx4 v[26:29], v4, s[0:3], 0 offen
	v_mov_b64_e32 v[102:103], v[32:33]
	buffer_load_dwordx4 v[30:33], v5, s[0:3], 0 offen
	v_mov_b64_e32 v[106:107], v[36:37]
	buffer_load_dwordx4 v[34:37], v6, s[0:3], 0 offen
	buffer_load_dwordx4 v[38:41], v7, s[0:3], 0 offen
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s0, s4
	buffer_load_dwordx4 v[42:45], v8, s[0:3], 0 offen
	buffer_load_dwordx4 v[46:49], v9, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[50:53], v50, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[54:57], v54, s[0:3], 0 offen
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	s_waitcnt vmcnt(11)
	v_cndmask_b32_e32 v59, 0, v13, vcc
	v_cndmask_b32_e32 v58, 0, v12, vcc
	v_cndmask_b32_e32 v63, 0, v11, vcc
	v_cndmask_b32_e32 v62, 0, v10, vcc
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	buffer_load_dwordx4 v[10:13], v70, s[0:3], 0 offen
	v_accvgpr_read_b32 v0, a127
	v_accvgpr_read_b32 v1, a128
	v_accvgpr_read_b32 v2, a129
	v_accvgpr_read_b32 v3, a130
	v_accvgpr_read_b32 v4, a131
	v_accvgpr_read_b32 v5, a132
	v_accvgpr_read_b32 v254, a138
	v_accvgpr_read_b32 v255, a137
	.loc	1 89 9                          ; 00_naive_mfma.py:89:9
	s_add_u32 s11, s11, 0x80
	s_addc_u32 s12, s12, 0
	.loc	1 90 9                          ; 00_naive_mfma.py:90:9
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_add_i32 s17, s17, -1
	s_sub_i32 s10, s10, 64
	s_cmp_lg_u32 s17, 0
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	s_waitcnt vmcnt(11)
	v_cndmask_b32_e32 v65, 0, v15, vcc
	v_cndmask_b32_e32 v64, 0, v14, vcc
	s_waitcnt vmcnt(10)
	v_cndmask_b32_e32 v67, 0, v21, vcc
	v_cndmask_b32_e32 v66, 0, v20, vcc
	s_waitcnt vmcnt(9)
	v_cndmask_b32_e32 v21, 0, v23, vcc
	v_cndmask_b32_e32 v20, 0, v22, vcc
	s_waitcnt vmcnt(8)
	v_cndmask_b32_e32 v23, 0, v29, vcc
	v_cndmask_b32_e32 v22, 0, v28, vcc
	s_waitcnt vmcnt(7)
	v_cndmask_b32_e32 v29, 0, v31, vcc
	v_cndmask_b32_e32 v28, 0, v30, vcc
	s_waitcnt vmcnt(6)
	v_cndmask_b32_e32 v31, 0, v37, vcc
	v_cndmask_b32_e32 v30, 0, v36, vcc
	s_waitcnt vmcnt(5)
	v_cndmask_b32_e32 v37, 0, v39, vcc
	v_cndmask_b32_e32 v36, 0, v38, vcc
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	s_waitcnt vmcnt(4)
	v_cndmask_b32_e32 v39, 0, v45, vcc
	v_cndmask_b32_e32 v38, 0, v44, vcc
	s_waitcnt vmcnt(3)
	v_cndmask_b32_e32 v45, 0, v47, vcc
	v_cndmask_b32_e32 v44, 0, v46, vcc
	s_waitcnt vmcnt(2)
	v_cndmask_b32_e32 v47, 0, v53, vcc
	v_cndmask_b32_e32 v46, 0, v52, vcc
	s_waitcnt vmcnt(1)
	v_cndmask_b32_e32 v53, 0, v55, vcc
	v_cndmask_b32_e32 v52, 0, v54, vcc
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v55, 0, v13, vcc
	v_cndmask_b32_e32 v54, 0, v12, vcc
	v_cndmask_b32_e32 v71, 0, v11, vcc
	v_cndmask_b32_e32 v70, 0, v10, vcc
	buffer_load_dwordx4 v[10:13], v72, s[0:3], 0 offen
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	v_cndmask_b32_e32 v69, 0, v25, vcc
	v_cndmask_b32_e32 v68, 0, v24, vcc
	v_cndmask_b32_e32 v25, 0, v33, vcc
	v_cndmask_b32_e32 v24, 0, v32, vcc
	v_cndmask_b32_e32 v33, 0, v41, vcc
	v_cndmask_b32_e32 v32, 0, v40, vcc
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	v_cndmask_b32_e32 v41, 0, v49, vcc
	v_cndmask_b32_e32 v40, 0, v48, vcc
	v_cndmask_b32_e32 v49, 0, v57, vcc
	v_cndmask_b32_e32 v48, 0, v56, vcc
	.loc	1 74 14                         ; 00_naive_mfma.py:74:14
	v_cndmask_b32_e32 v61, 0, v17, vcc
	v_cndmask_b32_e32 v60, 0, v16, vcc
	v_cndmask_b32_e32 v19, 0, v19, vcc
	v_cndmask_b32_e32 v18, 0, v18, vcc
	v_cndmask_b32_e32 v27, 0, v27, vcc
	v_cndmask_b32_e32 v26, 0, v26, vcc
	v_cndmask_b32_e32 v35, 0, v35, vcc
	v_cndmask_b32_e32 v34, 0, v34, vcc
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	v_cndmask_b32_e32 v43, 0, v43, vcc
	v_cndmask_b32_e32 v42, 0, v42, vcc
	v_cndmask_b32_e32 v51, 0, v51, vcc
	v_cndmask_b32_e32 v50, 0, v50, vcc
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v57, 0, v13, vcc
	v_cndmask_b32_e32 v56, 0, v12, vcc
	buffer_load_dwordx4 v[12:15], v74, s[0:3], 0 offen
	v_cndmask_b32_e32 v73, 0, v11, vcc
	buffer_load_dwordx4 v[74:77], v75, s[0:3], 0 offen
	.loc	1 81 13                         ; 00_naive_mfma.py:81:13
	s_barrier
	ds_write_b128 v0, v[62:65]
	ds_write_b128 v1, v[58:61]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[60:63], v2
	ds_read_b128 a[52:55], v3
	ds_read_b128 a[28:31], v4
	ds_read_b128 a[4:7], v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[18:21]
	ds_write_b128 v1, v[66:69]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[84:87], v2
	ds_read_b128 a[56:59], v3
	ds_read_b128 a[32:35], v4
	ds_read_b128 a[8:11], v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[26:29]
	ds_write_b128 v1, v[22:25]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[88:91], v2
	ds_read_b128 a[60:63], v3
	ds_read_b128 a[36:39], v4
	ds_read_b128 a[12:15], v5
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[34:37]
	ds_write_b128 v1, v[30:33]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[92:95], v2
	ds_read_b128 a[68:71], v3
	ds_read_b128 a[40:43], v4
	ds_read_b128 a[0:3], v5
	v_accvgpr_read_b32 v0, a133
	v_accvgpr_read_b32 v1, a134
	v_accvgpr_read_b32 v4, a135
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[42:45]
	ds_write_b128 v1, v[38:41]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[96:99], v4
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	v_cndmask_b32_e32 v72, 0, v10, vcc
	v_accvgpr_read_b32 v5, a136
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	ds_read_b128 a[72:75], v5
	ds_read_b128 a[44:47], v254
	ds_read_b128 a[16:19], v255
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[50:53]
	ds_write_b128 v1, v[46:49]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[100:103], v4
	ds_read_b128 a[76:79], v5
	ds_read_b128 a[48:51], v254
	ds_read_b128 a[20:23], v255
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b128 v0, v[70:73]
	ds_write_b128 v1, v[54:57]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[104:107], v4
	ds_read_b128 a[80:83], v5
	ds_read_b128 a[64:67], v254
	ds_read_b128 a[24:27], v255
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 84 15                         ; 00_naive_mfma.py:84:15
	v_mfma_f32_16x16x16_f16 v[6:9], a[96:97], v[60:61], v[78:81]
	.loc	1 77 14                         ; 00_naive_mfma.py:77:14
	s_waitcnt vmcnt(1)
	v_cndmask_b32_e32 v11, 0, v15, vcc
	v_cndmask_b32_e32 v10, 0, v14, vcc
	v_cndmask_b32_e32 v15, 0, v13, vcc
	v_cndmask_b32_e32 v14, 0, v12, vcc
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v17, 0, v75, vcc
	v_cndmask_b32_e32 v16, 0, v74, vcc
	v_cndmask_b32_e32 v13, 0, v77, vcc
	v_cndmask_b32_e32 v12, 0, v76, vcc
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	ds_write_b128 v0, v[14:17]
	ds_write_b128 v1, v[10:13]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[78:81], v4
	.loc	1 84 15                         ; 00_naive_mfma.py:84:15
	v_mfma_f32_16x16x16_f16 v[0:3], a[106:107], v[60:61], v[82:85]
	s_nop 2
	v_accvgpr_read_b32 v82, a184
	v_accvgpr_read_b32 v83, a185
	v_accvgpr_read_b32 v84, a186
	v_accvgpr_read_b32 v85, a187
	v_mfma_f32_16x16x16_f16 v[22:25], a[104:105], v[60:61], v[86:89]
	s_nop 0
	v_mfma_f32_16x16x16_f16 v[86:89], a[102:103], a[84:85], v[82:85]
	s_nop 2
	v_accvgpr_read_b32 v82, a216
	v_accvgpr_read_b32 v83, a217
	v_accvgpr_read_b32 v84, a218
	v_accvgpr_read_b32 v85, a219
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[30:33], v[80:81], v[60:61], v[94:97]
	v_mfma_f32_16x16x16_f16 v[94:97], a[106:107], a[84:85], v[110:113]
	v_mfma_f32_16x16x16_f16 v[110:113], a[98:99], a[86:87], v[126:129]
	v_mfma_f32_16x16x16_f16 v[126:129], a[106:107], a[86:87], v[142:145]
	v_mfma_f32_16x16x16_f16 v[142:145], a[98:99], a[88:89], v[158:161]
	v_mfma_f32_16x16x16_f16 v[158:161], a[106:107], a[88:89], v[174:177]
	v_mfma_f32_16x16x16_f16 v[174:177], a[98:99], a[90:91], v[190:193]
	v_mfma_f32_16x16x16_f16 v[190:193], a[106:107], a[90:91], v[82:85]
	s_nop 2
	v_accvgpr_read_b32 v82, a208
	v_accvgpr_read_b32 v83, a209
	v_accvgpr_read_b32 v84, a210
	v_accvgpr_read_b32 v85, a211
	v_mfma_f32_16x16x16_f16 v[42:45], a[100:101], v[62:63], v[106:109]
	v_mfma_f32_16x16x16_f16 v[106:109], a[96:97], a[86:87], v[122:125]
	v_mfma_f32_16x16x16_f16 v[122:125], a[104:105], a[86:87], v[138:141]
	v_mfma_f32_16x16x16_f16 v[138:141], a[96:97], a[88:89], v[154:157]
	v_mfma_f32_16x16x16_f16 v[154:157], a[104:105], a[88:89], v[170:173]
	v_mfma_f32_16x16x16_f16 v[170:173], a[96:97], a[90:91], v[186:189]
	v_mfma_f32_16x16x16_f16 v[186:189], a[104:105], a[90:91], v[202:205]
	v_mfma_f32_16x16x16_f16 v[202:205], a[96:97], a[92:93], v[218:221]
	v_mfma_f32_16x16x16_f16 v[218:221], a[102:103], a[92:93], v[82:85]
	s_nop 2
	v_accvgpr_read_b32 v82, a212
	v_accvgpr_read_b32 v83, a213
	v_accvgpr_read_b32 v84, a214
	v_accvgpr_read_b32 v85, a215
	v_mfma_f32_16x16x16_f16 v[34:37], a[96:97], v[62:63], v[98:101]
	v_mfma_f32_16x16x16_f16 v[98:101], v[78:79], a[84:85], v[114:117]
	v_mfma_f32_16x16x16_f16 v[114:117], a[100:101], a[86:87], v[130:133]
	v_mfma_f32_16x16x16_f16 v[130:133], v[78:79], a[86:87], v[146:149]
	v_mfma_f32_16x16x16_f16 v[146:149], a[100:101], a[88:89], v[162:165]
	v_mfma_f32_16x16x16_f16 v[162:165], v[78:79], a[88:89], v[178:181]
	v_mfma_f32_16x16x16_f16 v[178:181], a[100:101], a[90:91], v[194:197]
	v_mfma_f32_16x16x16_f16 v[194:197], v[78:79], a[90:91], v[210:213]
	v_mfma_f32_16x16x16_f16 v[210:213], a[98:99], a[92:93], v[222:225]
	v_mfma_f32_16x16x16_f16 v[222:225], a[104:105], a[92:93], v[82:85]
	s_nop 2
	v_accvgpr_read_b32 v82, a196
	v_accvgpr_read_b32 v83, a197
	v_accvgpr_read_b32 v84, a198
	v_accvgpr_read_b32 v85, a199
	v_mfma_f32_16x16x16_f16 v[38:41], a[98:99], v[62:63], v[102:105]
	v_mfma_f32_16x16x16_f16 v[102:105], v[80:81], a[84:85], v[118:121]
	v_mfma_f32_16x16x16_f16 v[118:121], a[102:103], a[86:87], v[134:137]
	v_mfma_f32_16x16x16_f16 v[134:137], v[80:81], a[86:87], v[150:153]
	v_mfma_f32_16x16x16_f16 v[150:153], a[102:103], a[88:89], v[166:169]
	v_mfma_f32_16x16x16_f16 v[166:169], v[80:81], a[88:89], v[182:185]
	v_mfma_f32_16x16x16_f16 v[182:185], a[102:103], a[90:91], v[198:201]
	v_mfma_f32_16x16x16_f16 v[198:201], v[80:81], a[90:91], v[214:217]
	v_accvgpr_mov_b32 a88, a200
	v_accvgpr_mov_b32 a89, a201
	v_accvgpr_mov_b32 a90, a202
	v_mfma_f32_16x16x16_f16 v[214:217], a[100:101], a[92:93], v[226:229]
	v_accvgpr_mov_b32 a91, a203
	v_mfma_f32_16x16x16_f16 v[226:229], a[106:107], a[92:93], v[238:241]
	v_mfma_f32_16x16x16_f16 v[238:241], v[78:79], a[92:93], v[242:245]
	v_mfma_f32_16x16x16_f16 v[242:245], v[80:81], a[92:93], v[246:249]
	v_mfma_f32_16x16x16_f16 v[246:249], a[96:97], a[94:95], v[250:253]
	v_mfma_f32_16x16x16_f16 v[250:253], a[98:99], a[94:95], v[82:85]
	v_mfma_f32_16x16x16_f16 v[82:85], a[104:105], a[94:95], v[234:237]
	v_mfma_f32_16x16x16_f16 a[220:223], a[102:103], v[60:61], a[148:151]
	v_mfma_f32_16x16x16_f16 a[148:151], a[98:99], v[60:61], a[140:143]
	v_mfma_f32_16x16x16_f16 a[244:247], a[96:97], a[84:85], a[172:175]
	v_mfma_f32_16x16x16_f16 a[248:251], a[98:99], a[84:85], a[176:179]
	v_accvgpr_mov_b32 a96, a204
	v_accvgpr_mov_b32 a97, a205
	v_accvgpr_mov_b32 a98, a206
	v_accvgpr_mov_b32 a99, a207
	v_mfma_f32_16x16x16_f16 a[144:147], a[100:101], v[60:61], a[144:147]
	v_mfma_f32_16x16x16_f16 a[224:227], a[102:103], v[62:63], a[152:155]
	v_mfma_f32_16x16x16_f16 a[252:255], a[100:101], a[84:85], a[180:183]
	v_mfma_f32_16x16x16_f16 a[88:91], a[100:101], a[94:95], a[88:91]
	v_mfma_f32_16x16x16_f16 a[96:99], a[102:103], a[94:95], a[96:99]
	v_accvgpr_write_b32 a103, v85
	v_accvgpr_write_b32 a102, v84
	v_accvgpr_write_b32 a101, v83
	v_accvgpr_write_b32 a100, v82
	v_mfma_f32_16x16x16_f16 v[82:85], a[106:107], a[94:95], v[230:233]
	v_mfma_f32_16x16x16_f16 a[188:191], a[104:105], a[84:85], a[188:191]
	v_accvgpr_mov_b32 a84, a192
	v_accvgpr_mov_b32 a85, a193
	v_accvgpr_mov_b32 a86, a194
	v_mfma_f32_16x16x16_f16 a[228:231], a[104:105], v[62:63], a[156:159]
	v_accvgpr_mov_b32 a87, a195
	v_mfma_f32_16x16x16_f16 a[232:235], a[106:107], v[62:63], a[160:163]
	v_accvgpr_write_b32 a107, v85
	v_accvgpr_write_b32 a106, v84
	v_accvgpr_write_b32 a105, v83
	v_mfma_f32_16x16x16_f16 v[26:29], v[78:79], v[60:61], v[90:93]
	v_accvgpr_write_b32 a104, v82
	v_mfma_f32_16x16x16_f16 a[236:239], v[78:79], v[62:63], a[164:167]
	v_mfma_f32_16x16x16_f16 a[240:243], v[80:81], v[62:63], a[168:171]
	v_mfma_f32_16x16x16_f16 v[82:85], v[78:79], a[94:95], v[206:209]
	v_mfma_f32_16x16x16_f16 a[92:95], v[80:81], a[94:95], a[84:87]
	v_mfma_f32_16x16x16_f16 v[78:81], a[72:73], a[52:53], v[6:9]
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_nop 2
	ds_read_b128 v[4:7], v5
	.loc	1 84 15                         ; 00_naive_mfma.py:84:15
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[12:15], v[4:5], a[52:53], v[26:29]
	v_accvgpr_mov_b32 a195, a95
	v_accvgpr_mov_b32 a194, a94
	v_accvgpr_mov_b32 a193, a93
	v_mfma_f32_16x16x16_f16 v[0:3], a[82:83], a[52:53], v[0:3]
	v_accvgpr_mov_b32 a192, a92
	v_mfma_f32_16x16x16_f16 a[152:155], a[76:77], a[52:53], a[144:147]
	s_nop 2
	v_accvgpr_write_b32 a147, v15
	v_accvgpr_write_b32 a146, v14
	v_accvgpr_write_b32 a145, v13
	v_accvgpr_write_b32 a144, v12
	v_accvgpr_read_b32 v12, a88
	v_accvgpr_read_b32 v13, a89
	v_accvgpr_read_b32 v14, a90
	v_accvgpr_read_b32 v15, a91
	v_mfma_f32_16x16x16_f16 a[176:179], a[72:73], a[56:57], a[244:247]
	v_mfma_f32_16x16x16_f16 a[180:183], a[74:75], a[56:57], a[248:251]
	v_mfma_f32_16x16x16_f16 a[184:187], a[76:77], a[56:57], a[252:255]
	v_mfma_f32_16x16x16_f16 v[74:77], a[78:79], a[56:57], v[86:89]
	v_mfma_f32_16x16x16_f16 a[188:191], a[80:81], a[56:57], a[188:191]
	v_mfma_f32_16x16x16_f16 v[86:89], a[82:83], a[56:57], v[94:97]
	v_mfma_f32_16x16x16_f16 v[90:93], v[4:5], a[56:57], v[98:101]
	v_mfma_f32_16x16x16_f16 v[94:97], v[6:7], a[56:57], v[102:105]
	v_mfma_f32_16x16x16_f16 v[98:101], a[72:73], a[58:59], v[106:109]
	v_mfma_f32_16x16x16_f16 v[102:105], a[74:75], a[58:59], v[110:113]
	v_mfma_f32_16x16x16_f16 v[106:109], a[76:77], a[58:59], v[114:117]
	v_mfma_f32_16x16x16_f16 v[110:113], a[78:79], a[58:59], v[118:121]
	v_mfma_f32_16x16x16_f16 v[114:117], a[80:81], a[58:59], v[122:125]
	v_mfma_f32_16x16x16_f16 v[118:121], a[82:83], a[58:59], v[126:129]
	v_mfma_f32_16x16x16_f16 v[122:125], v[4:5], a[58:59], v[130:133]
	v_mfma_f32_16x16x16_f16 v[126:129], v[6:7], a[58:59], v[134:137]
	v_accvgpr_write_b32 a59, v3
	v_accvgpr_write_b32 a58, v2
	v_accvgpr_write_b32 a57, v1
	v_mfma_f32_16x16x16_f16 v[234:237], a[76:77], a[70:71], v[12:15]
	v_accvgpr_write_b32 a56, v0
	v_accvgpr_read_b32 v0, a144
	v_accvgpr_read_b32 v1, a145
	v_accvgpr_read_b32 v12, a96
	v_accvgpr_read_b32 v13, a97
	v_accvgpr_read_b32 v14, a98
	v_accvgpr_read_b32 v15, a99
	v_mfma_f32_16x16x16_f16 v[206:209], a[78:79], a[68:69], v[218:221]
	v_accvgpr_read_b32 v2, a146
	v_accvgpr_read_b32 v3, a147
	v_mfma_f32_16x16x16_f16 v[218:221], v[4:5], a[68:69], v[238:241]
	v_mfma_f32_16x16x16_f16 v[238:241], a[78:79], a[70:71], v[12:15]
	s_nop 2
	v_accvgpr_read_b32 v12, a100
	v_mfma_f32_16x16x16_f16 a[56:59], a[66:67], a[28:29], a[56:59]
	v_accvgpr_read_b32 v13, a101
	v_accvgpr_read_b32 v14, a102
	v_accvgpr_read_b32 v15, a103
	v_mfma_f32_16x16x16_f16 a[148:151], a[74:75], a[52:53], a[148:151]
	v_mfma_f32_16x16x16_f16 a[84:87], a[78:79], a[52:53], a[220:223]
	s_nop 1
	v_accvgpr_read_b32 v8, a56
	v_accvgpr_read_b32 v9, a57
	v_accvgpr_read_b32 v10, a58
	v_mfma_f32_16x16x16_f16 v[22:25], a[80:81], a[52:53], v[22:25]
	v_accvgpr_read_b32 v11, a59
	v_mfma_f32_16x16x16_f16 v[26:29], v[6:7], a[52:53], v[30:33]
	v_mfma_f32_16x16x16_f16 v[30:33], a[72:73], a[54:55], v[34:37]
	v_mfma_f32_16x16x16_f16 v[34:37], a[74:75], a[54:55], v[38:41]
	v_mfma_f32_16x16x16_f16 v[38:41], a[76:77], a[54:55], v[42:45]
	v_mfma_f32_16x16x16_f16 a[156:159], a[78:79], a[54:55], a[224:227]
	v_mfma_f32_16x16x16_f16 a[160:163], a[80:81], a[54:55], a[228:231]
	v_mfma_f32_16x16x16_f16 a[164:167], a[82:83], a[54:55], a[232:235]
	v_mfma_f32_16x16x16_f16 a[168:171], v[4:5], a[54:55], a[236:239]
	v_mfma_f32_16x16x16_f16 a[172:175], v[6:7], a[54:55], a[240:243]
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	ds_read_b128 a[52:55], v254
	.loc	1 84 15                         ; 00_naive_mfma.py:84:15
	v_mfma_f32_16x16x16_f16 v[134:137], a[74:75], a[60:61], v[142:145]
	v_mfma_f32_16x16x16_f16 v[142:145], a[78:79], a[60:61], v[150:153]
	v_mfma_f32_16x16x16_f16 v[150:153], a[82:83], a[60:61], v[158:161]
	v_mfma_f32_16x16x16_f16 v[158:161], v[6:7], a[60:61], v[166:169]
	v_mfma_f32_16x16x16_f16 v[166:169], a[74:75], a[62:63], v[174:177]
	v_mfma_f32_16x16x16_f16 v[174:177], a[78:79], a[62:63], v[182:185]
	v_mfma_f32_16x16x16_f16 v[182:185], a[82:83], a[62:63], v[190:193]
	v_mfma_f32_16x16x16_f16 v[190:193], v[6:7], a[62:63], v[198:201]
	v_mfma_f32_16x16x16_f16 v[198:201], a[74:75], a[68:69], v[210:213]
	v_mfma_f32_16x16x16_f16 v[210:213], a[80:81], a[68:69], v[222:225]
	v_mfma_f32_16x16x16_f16 v[222:225], v[6:7], a[68:69], v[242:245]
	v_mfma_f32_16x16x16_f16 v[242:245], a[80:81], a[70:71], v[12:15]
	s_nop 2
	v_accvgpr_read_b32 v12, a104
	v_accvgpr_read_b32 v13, a105
	v_accvgpr_read_b32 v14, a106
	v_accvgpr_read_b32 v15, a107
	v_mfma_f32_16x16x16_f16 v[130:133], a[72:73], a[60:61], v[138:141]
	v_mfma_f32_16x16x16_f16 v[138:141], a[76:77], a[60:61], v[146:149]
	v_mfma_f32_16x16x16_f16 v[146:149], a[80:81], a[60:61], v[154:157]
	v_mfma_f32_16x16x16_f16 v[154:157], v[4:5], a[60:61], v[162:165]
	v_mfma_f32_16x16x16_f16 v[162:165], a[72:73], a[62:63], v[170:173]
	v_mfma_f32_16x16x16_f16 v[170:173], a[76:77], a[62:63], v[178:181]
	v_mfma_f32_16x16x16_f16 v[178:181], a[80:81], a[62:63], v[186:189]
	v_mfma_f32_16x16x16_f16 v[186:189], v[4:5], a[62:63], v[194:197]
	v_mfma_f32_16x16x16_f16 v[194:197], a[72:73], a[68:69], v[202:205]
	v_mfma_f32_16x16x16_f16 v[202:205], a[76:77], a[68:69], v[214:217]
	v_mfma_f32_16x16x16_f16 v[214:217], a[82:83], a[68:69], v[226:229]
	v_mfma_f32_16x16x16_f16 v[226:229], a[72:73], a[70:71], v[246:249]
	v_mfma_f32_16x16x16_f16 v[246:249], a[82:83], a[70:71], v[12:15]
	s_nop 2
	v_mov_b64_e32 v[12:13], v[82:83]
	v_mov_b64_e32 v[14:15], v[84:85]
	v_mfma_f32_16x16x16_f16 v[230:233], a[74:75], a[70:71], v[250:253]
	s_nop 0
	v_mfma_f32_16x16x16_f16 v[250:253], v[4:5], a[70:71], v[12:15]
	v_mfma_f32_16x16x16_f16 v[12:15], a[26:27], a[4:5], v[8:11]
	.loc	1 82 13                         ; 00_naive_mfma.py:82:13
	s_nop 2
	ds_read_b128 v[8:11], v255
	.loc	1 84 15                         ; 00_naive_mfma.py:84:15
	v_mfma_f32_16x16x16_f16 a[192:195], v[6:7], a[70:71], a[192:195]
	v_mfma_f32_16x16x16_f16 a[140:143], a[46:47], a[28:29], a[148:151]
	s_nop 2
	v_accvgpr_mov_b32 a151, a87
	v_mfma_f32_16x16x16_f16 v[20:23], a[64:65], a[28:29], v[22:25]
	v_accvgpr_mov_b32 a150, a86
	v_accvgpr_mov_b32 a149, a85
	v_accvgpr_mov_b32 a148, a84
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 v[0:3], a[52:53], a[28:29], v[0:3]
	v_mfma_f32_16x16x16_f16 v[72:75], a[50:51], a[32:33], v[74:77]
	v_mfma_f32_16x16x16_f16 v[78:81], a[44:45], a[28:29], v[78:81]
	v_mfma_f32_16x16x16_f16 a[60:63], a[48:49], a[28:29], a[152:155]
	v_mfma_f32_16x16x16_f16 a[148:151], a[50:51], a[28:29], a[148:151]
	v_mfma_f32_16x16x16_f16 v[24:27], a[54:55], a[28:29], v[26:29]
	v_mfma_f32_16x16x16_f16 v[28:31], a[44:45], a[30:31], v[30:33]
	v_mfma_f32_16x16x16_f16 v[32:35], a[46:47], a[30:31], v[34:37]
	v_mfma_f32_16x16x16_f16 v[36:39], a[48:49], a[30:31], v[38:41]
	v_mfma_f32_16x16x16_f16 a[152:155], a[50:51], a[30:31], a[156:159]
	v_mfma_f32_16x16x16_f16 a[156:159], a[64:65], a[30:31], a[160:163]
	v_mfma_f32_16x16x16_f16 a[160:163], a[66:67], a[30:31], a[164:167]
	v_mfma_f32_16x16x16_f16 a[164:167], a[52:53], a[30:31], a[168:171]
	v_mfma_f32_16x16x16_f16 a[168:171], a[54:55], a[30:31], a[172:175]
	v_mfma_f32_16x16x16_f16 a[172:175], a[44:45], a[32:33], a[176:179]
	v_mfma_f32_16x16x16_f16 a[176:179], a[46:47], a[32:33], a[180:183]
	v_mfma_f32_16x16x16_f16 a[180:183], a[48:49], a[32:33], a[184:187]
	v_mfma_f32_16x16x16_f16 a[188:191], a[64:65], a[32:33], a[188:191]
	v_mfma_f32_16x16x16_f16 v[86:89], a[66:67], a[32:33], v[86:89]
	v_mfma_f32_16x16x16_f16 v[90:93], a[52:53], a[32:33], v[90:93]
	v_mfma_f32_16x16x16_f16 v[94:97], a[54:55], a[32:33], v[94:97]
	v_mfma_f32_16x16x16_f16 v[98:101], a[44:45], a[34:35], v[98:101]
	v_mfma_f32_16x16x16_f16 v[102:105], a[46:47], a[34:35], v[102:105]
	v_mfma_f32_16x16x16_f16 v[106:109], a[48:49], a[34:35], v[106:109]
	v_mfma_f32_16x16x16_f16 v[110:113], a[50:51], a[34:35], v[110:113]
	v_mfma_f32_16x16x16_f16 v[114:117], a[64:65], a[34:35], v[114:117]
	v_mfma_f32_16x16x16_f16 v[118:121], a[66:67], a[34:35], v[118:121]
	v_mfma_f32_16x16x16_f16 v[122:125], a[52:53], a[34:35], v[122:125]
	v_mfma_f32_16x16x16_f16 v[126:129], a[54:55], a[34:35], v[126:129]
	v_mfma_f32_16x16x16_f16 v[130:133], a[44:45], a[36:37], v[130:133]
	v_mfma_f32_16x16x16_f16 v[134:137], a[46:47], a[36:37], v[134:137]
	v_mfma_f32_16x16x16_f16 v[138:141], a[48:49], a[36:37], v[138:141]
	v_mfma_f32_16x16x16_f16 v[142:145], a[50:51], a[36:37], v[142:145]
	v_mfma_f32_16x16x16_f16 v[146:149], a[64:65], a[36:37], v[146:149]
	v_mfma_f32_16x16x16_f16 v[150:153], a[66:67], a[36:37], v[150:153]
	v_mfma_f32_16x16x16_f16 v[154:157], a[52:53], a[36:37], v[154:157]
	v_mfma_f32_16x16x16_f16 v[158:161], a[54:55], a[36:37], v[158:161]
	v_mfma_f32_16x16x16_f16 v[162:165], a[44:45], a[38:39], v[162:165]
	v_mfma_f32_16x16x16_f16 v[166:169], a[46:47], a[38:39], v[166:169]
	v_mfma_f32_16x16x16_f16 v[170:173], a[48:49], a[38:39], v[170:173]
	v_mfma_f32_16x16x16_f16 v[174:177], a[50:51], a[38:39], v[174:177]
	v_mfma_f32_16x16x16_f16 v[178:181], a[64:65], a[38:39], v[178:181]
	v_mfma_f32_16x16x16_f16 v[182:185], a[66:67], a[38:39], v[182:185]
	v_mfma_f32_16x16x16_f16 v[186:189], a[52:53], a[38:39], v[186:189]
	v_mfma_f32_16x16x16_f16 v[190:193], a[54:55], a[38:39], v[190:193]
	v_mfma_f32_16x16x16_f16 v[194:197], a[44:45], a[40:41], v[194:197]
	v_mfma_f32_16x16x16_f16 v[198:201], a[46:47], a[40:41], v[198:201]
	v_mfma_f32_16x16x16_f16 v[202:205], a[48:49], a[40:41], v[202:205]
	v_mfma_f32_16x16x16_f16 v[206:209], a[50:51], a[40:41], v[206:209]
	v_mfma_f32_16x16x16_f16 v[210:213], a[64:65], a[40:41], v[210:213]
	v_mfma_f32_16x16x16_f16 v[214:217], a[66:67], a[40:41], v[214:217]
	v_mfma_f32_16x16x16_f16 v[218:221], a[52:53], a[40:41], v[218:221]
	v_mfma_f32_16x16x16_f16 v[222:225], a[54:55], a[40:41], v[222:225]
	v_mfma_f32_16x16x16_f16 v[226:229], a[44:45], a[42:43], v[226:229]
	v_mfma_f32_16x16x16_f16 v[230:233], a[46:47], a[42:43], v[230:233]
	v_mfma_f32_16x16x16_f16 v[234:237], a[48:49], a[42:43], v[234:237]
	v_mfma_f32_16x16x16_f16 v[238:241], a[50:51], a[42:43], v[238:241]
	v_mfma_f32_16x16x16_f16 v[242:245], a[64:65], a[42:43], v[242:245]
	v_mfma_f32_16x16x16_f16 v[246:249], a[66:67], a[42:43], v[246:249]
	v_mfma_f32_16x16x16_f16 v[250:253], a[52:53], a[42:43], v[250:253]
	v_mfma_f32_16x16x16_f16 a[192:195], a[54:55], a[42:43], a[192:195]
	v_mfma_f32_16x16x16_f16 v[16:19], a[24:25], a[4:5], v[20:23]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 v[20:23], v[8:9], a[4:5], v[0:3]
	v_mfma_f32_16x16x16_f16 v[0:3], a[22:23], a[8:9], v[72:75]
	v_mfma_f32_16x16x16_f16 v[78:81], a[16:17], a[4:5], v[78:81]
	v_mfma_f32_16x16x16_f16 a[140:143], a[18:19], a[4:5], a[140:143]
	s_nop 4
	v_accvgpr_write_b32 a187, v3
	v_accvgpr_write_b32 a186, v2
	v_accvgpr_write_b32 a185, v1
	v_mfma_f32_16x16x16_f16 a[144:147], a[20:21], a[4:5], a[60:63]
	v_accvgpr_write_b32 a184, v0
	v_mfma_f32_16x16x16_f16 a[148:151], a[22:23], a[4:5], a[148:151]
	v_mfma_f32_16x16x16_f16 v[24:27], v[10:11], a[4:5], v[24:27]
	v_mfma_f32_16x16x16_f16 v[28:31], a[16:17], a[6:7], v[28:31]
	v_mfma_f32_16x16x16_f16 v[32:35], a[18:19], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_f16 v[36:39], a[20:21], a[6:7], v[36:39]
	v_mfma_f32_16x16x16_f16 a[152:155], a[22:23], a[6:7], a[152:155]
	v_mfma_f32_16x16x16_f16 a[156:159], a[24:25], a[6:7], a[156:159]
	v_mfma_f32_16x16x16_f16 a[160:163], a[26:27], a[6:7], a[160:163]
	v_mfma_f32_16x16x16_f16 a[164:167], v[8:9], a[6:7], a[164:167]
	v_mfma_f32_16x16x16_f16 a[168:171], v[10:11], a[6:7], a[168:171]
	v_mfma_f32_16x16x16_f16 a[172:175], a[16:17], a[8:9], a[172:175]
	v_mfma_f32_16x16x16_f16 a[176:179], a[18:19], a[8:9], a[176:179]
	v_mfma_f32_16x16x16_f16 a[180:183], a[20:21], a[8:9], a[180:183]
	v_mfma_f32_16x16x16_f16 a[188:191], a[24:25], a[8:9], a[188:191]
	v_mfma_f32_16x16x16_f16 v[86:89], a[26:27], a[8:9], v[86:89]
	v_mfma_f32_16x16x16_f16 v[90:93], v[8:9], a[8:9], v[90:93]
	v_mfma_f32_16x16x16_f16 v[94:97], v[10:11], a[8:9], v[94:97]
	v_mfma_f32_16x16x16_f16 v[98:101], a[16:17], a[10:11], v[98:101]
	v_mfma_f32_16x16x16_f16 v[102:105], a[18:19], a[10:11], v[102:105]
	v_mfma_f32_16x16x16_f16 v[106:109], a[20:21], a[10:11], v[106:109]
	v_mfma_f32_16x16x16_f16 v[110:113], a[22:23], a[10:11], v[110:113]
	v_mfma_f32_16x16x16_f16 v[114:117], a[24:25], a[10:11], v[114:117]
	v_mfma_f32_16x16x16_f16 v[118:121], a[26:27], a[10:11], v[118:121]
	v_mfma_f32_16x16x16_f16 v[122:125], v[8:9], a[10:11], v[122:125]
	v_mfma_f32_16x16x16_f16 v[126:129], v[10:11], a[10:11], v[126:129]
	v_mfma_f32_16x16x16_f16 v[130:133], a[16:17], a[12:13], v[130:133]
	v_mfma_f32_16x16x16_f16 v[134:137], a[18:19], a[12:13], v[134:137]
	v_mfma_f32_16x16x16_f16 v[138:141], a[20:21], a[12:13], v[138:141]
	v_mfma_f32_16x16x16_f16 v[142:145], a[22:23], a[12:13], v[142:145]
	v_mfma_f32_16x16x16_f16 v[146:149], a[24:25], a[12:13], v[146:149]
	v_mfma_f32_16x16x16_f16 v[150:153], a[26:27], a[12:13], v[150:153]
	v_mfma_f32_16x16x16_f16 v[154:157], v[8:9], a[12:13], v[154:157]
	v_mfma_f32_16x16x16_f16 v[158:161], v[10:11], a[12:13], v[158:161]
	v_mfma_f32_16x16x16_f16 v[162:165], a[16:17], a[14:15], v[162:165]
	v_mfma_f32_16x16x16_f16 v[166:169], a[18:19], a[14:15], v[166:169]
	v_mfma_f32_16x16x16_f16 v[170:173], a[20:21], a[14:15], v[170:173]
	v_mfma_f32_16x16x16_f16 v[174:177], a[22:23], a[14:15], v[174:177]
	v_mfma_f32_16x16x16_f16 v[178:181], a[24:25], a[14:15], v[178:181]
	v_mfma_f32_16x16x16_f16 v[182:185], a[26:27], a[14:15], v[182:185]
	v_mfma_f32_16x16x16_f16 v[186:189], v[8:9], a[14:15], v[186:189]
	v_mfma_f32_16x16x16_f16 v[190:193], v[10:11], a[14:15], v[190:193]
	v_mfma_f32_16x16x16_f16 v[194:197], a[16:17], a[0:1], v[194:197]
	v_mfma_f32_16x16x16_f16 v[198:201], a[18:19], a[0:1], v[198:201]
	v_mfma_f32_16x16x16_f16 v[202:205], a[20:21], a[0:1], v[202:205]
	v_mfma_f32_16x16x16_f16 v[206:209], a[22:23], a[0:1], v[206:209]
	v_mfma_f32_16x16x16_f16 v[210:213], a[24:25], a[0:1], v[210:213]
	v_mfma_f32_16x16x16_f16 v[214:217], a[26:27], a[0:1], v[214:217]
	v_mfma_f32_16x16x16_f16 v[218:221], v[8:9], a[0:1], v[218:221]
	v_mfma_f32_16x16x16_f16 v[222:225], v[10:11], a[0:1], v[222:225]
	v_mfma_f32_16x16x16_f16 v[226:229], a[16:17], a[2:3], v[226:229]
	v_mfma_f32_16x16x16_f16 v[230:233], a[18:19], a[2:3], v[230:233]
	v_mfma_f32_16x16x16_f16 v[234:237], a[20:21], a[2:3], v[234:237]
	v_mfma_f32_16x16x16_f16 v[238:241], a[22:23], a[2:3], v[238:241]
	v_mfma_f32_16x16x16_f16 v[242:245], a[24:25], a[2:3], v[242:245]
	v_mfma_f32_16x16x16_f16 v[246:249], a[26:27], a[2:3], v[246:249]
	v_mfma_f32_16x16x16_f16 v[250:253], v[8:9], a[2:3], v[250:253]
	v_mfma_f32_16x16x16_f16 a[192:195], v[10:11], a[2:3], a[192:195]
	.loc	1 70 5                          ; 00_naive_mfma.py:70:5
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %._crit_edge.loopexit
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v0, v78
	v_accvgpr_read_b32 v8, a144
	v_accvgpr_read_b32 v9, a145
	v_accvgpr_read_b32 v10, a146
	v_accvgpr_write_b32 a37, v0
	v_cvt_f16_f32_e32 v0, v79
	v_accvgpr_read_b32 v11, a147
	v_cvt_f16_f32_e32 v74, v12
	v_cvt_f16_f32_e32 v68, v24
	v_accvgpr_write_b32 a34, v0
	v_cvt_f16_f32_e32 v0, v80
	v_cvt_f16_f32_e32 v83, v92
	v_cvt_f16_f32_e32 v71, v116
	v_cvt_f16_f32_e32 v92, v117
	v_accvgpr_write_b32 a0, v0
	v_cvt_f16_f32_e32 v0, v81
	v_cvt_f16_f32_e32 v81, v96
	v_cvt_f16_f32_e32 v78, v86
	v_cvt_f16_f32_e32 v86, v93
	v_accvgpr_write_b32 a1, v0
	v_accvgpr_read_b32 v0, a140
	v_cvt_f16_f32_e32 v0, v0
	v_accvgpr_read_b32 v1, a141
	v_accvgpr_read_b32 v2, a142
	v_accvgpr_read_b32 v3, a143
	v_accvgpr_write_b32 a55, v0
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v69, v120
	v_cvt_f16_f32_e32 v93, v121
	v_cvt_f16_f32_e32 v60, v95
	v_accvgpr_write_b32 a30, v0
	v_cvt_f16_f32_e32 v0, v2
	v_cvt_f16_f32_e32 v65, v128
	v_cvt_f16_f32_e32 v95, v129
	v_cvt_f16_f32_e32 v61, v136
	v_accvgpr_write_b32 a2, v0
	v_cvt_f16_f32_e32 v0, v3
	v_cvt_f16_f32_e32 v56, v99
	v_cvt_f16_f32_e32 v57, v144
	v_cvt_f16_f32_e32 v99, v145
	v_accvgpr_write_b32 a3, v0
	v_cvt_f16_f32_e32 v0, v8
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v71, v71, v92
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v84, v88
	v_cvt_f16_f32_e32 v88, v101
	v_accvgpr_write_b32 a22, v0
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v53, v152
	v_cvt_f16_f32_e32 v101, v153
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v69, v69, v93
	v_accvgpr_write_b32 a8, v0
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v0, v10
	v_cvt_f16_f32_e32 v52, v103
	v_cvt_f16_f32_e32 v49, v160
	v_cvt_f16_f32_e32 v103, v161
	v_accvgpr_write_b32 a4, v0
	v_cvt_f16_f32_e32 v0, v11
	v_accvgpr_read_b32 v8, a148
	v_accvgpr_read_b32 v9, a149
	v_accvgpr_read_b32 v10, a150
	v_accvgpr_write_b32 a5, v0
	v_cvt_f16_f32_e32 v0, v8
	v_accvgpr_read_b32 v11, a151
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v65, v65, v95
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v85, v89
	v_accvgpr_write_b32 a11, v0
	v_cvt_f16_f32_e32 v0, v9
	v_accvgpr_read_b32 v6, a168
	v_accvgpr_read_b32 v8, a170
	v_accvgpr_read_b32 v9, a171
	v_accvgpr_write_b32 a12, v0
	v_cvt_f16_f32_e32 v0, v10
	v_accvgpr_read_b32 v7, a169
	v_cvt_f16_f32_e32 v89, v105
	v_cvt_f16_f32_e32 v45, v168
	v_accvgpr_write_b32 a6, v0
	v_cvt_f16_f32_e32 v0, v11
	v_cvt_f16_f32_e32 v105, v169
	v_cvt_f16_f32_e32 v48, v107
	v_cvt_f16_f32_e32 v41, v176
	v_accvgpr_write_b32 a7, v0
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v107, v177
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v57, v57, v99
	v_pack_b32_f16 v53, v53, v101
	v_accvgpr_write_b32 a15, v0
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v0, v17
	v_cvt_f16_f32_e32 v64, v33
	v_cvt_f16_f32_e32 v44, v111
	v_cvt_f16_f32_e32 v33, v192
	v_accvgpr_write_b32 a49, v0
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v111, v193
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v49, v49, v103
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v66, v29
	v_accvgpr_write_b32 a9, v0
	v_cvt_f16_f32_e32 v0, v19
	v_cvt_f16_f32_e32 v29, v200
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v45, v45, v105
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v40, v115
	v_accvgpr_write_b32 a10, v0
	v_cvt_f16_f32_e32 v0, v13
	v_cvt_f16_f32_e32 v115, v209
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v41, v41, v107
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v117, v217
	v_accvgpr_write_b32 a52, v0
	v_cvt_f16_f32_e32 v0, v14
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v33, v33, v111
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v121, v233
	v_cvt_f16_f32_e32 v152, v167
	v_accvgpr_write_b32 a13, v0
	v_cvt_f16_f32_e32 v0, v15
	v_accvgpr_read_b32 v14, a180
	v_accvgpr_read_b32 v15, a181
	v_accvgpr_read_b32 v16, a182
	v_accvgpr_write_b32 a14, v0
	v_cvt_f16_f32_e32 v0, v20
	v_cvt_f16_f32_e32 v82, v16
	v_cvt_f16_f32_e32 v16, v87
	v_cvt_f16_f32_e32 v87, v97
	v_accvgpr_write_b32 a21, v0
	v_cvt_f16_f32_e32 v0, v21
	v_cvt_f16_f32_e32 v97, v137
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v81, v81, v87
	v_accvgpr_read_b32 v17, a183
	v_accvgpr_write_b32 a20, v0
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v0, v22
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v61, v61, v97
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v22, v37
	v_cvt_f16_f32_e32 v37, v184
	v_accvgpr_write_b32 a16, v0
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v21, v216
	v_cvt_f16_f32_e32 v80, v17
	v_cvt_f16_f32_e32 v17, v224
	v_accvgpr_write_b32 a17, v0
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v145, v194
	v_cvt_f16_f32_e32 v167, v195
	v_accvgpr_read_b32 v192, a192
	v_accvgpr_write_b32 a25, v0
	v_cvt_f16_f32_e32 v0, v26
	v_cvt_f16_f32_e32 v62, v94
	v_cvt_f16_f32_e32 v94, v125
	v_cvt_f16_f32_e32 v5, v248
	v_accvgpr_write_b32 a18, v0
	v_cvt_f16_f32_e32 v0, v27
	v_accvgpr_read_b32 v24, a184
	v_accvgpr_read_b32 v25, a185
	v_accvgpr_read_b32 v26, a186
	v_accvgpr_write_b32 a19, v0
	v_cvt_f16_f32_e32 v0, v28
	v_accvgpr_read_b32 v27, a187
	v_cvt_f16_f32_e32 v254, v27
	v_cvt_f16_f32_e32 v125, v249
	v_accvgpr_write_b32 a27, v0
	v_cvt_f16_f32_e32 v0, v30
	v_accvgpr_read_b32 v194, a194
	v_accvgpr_read_b32 v195, a195
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v21, v21, v117
	v_accvgpr_write_b32 a23, v0
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v58, v98
	v_cvt_f16_f32_e32 v79, v100
	v_cvt_f16_f32_e32 v54, v102
	v_accvgpr_write_b32 a24, v0
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v32, v123
	v_cvt_f16_f32_e32 v123, v241
	v_cvt_f16_f32_e32 v77, v104
	v_accvgpr_write_b32 a32, v0
	v_cvt_f16_f32_e32 v0, v34
	v_cvt_f16_f32_e32 v50, v106
	v_cvt_f16_f32_e32 v75, v108
	v_cvt_f16_f32_e32 v46, v110
	v_accvgpr_write_b32 a26, v0
	v_cvt_f16_f32_e32 v0, v35
	v_cvt_f16_f32_e32 v73, v112
	v_cvt_f16_f32_e32 v42, v114
	v_cvt_f16_f32_e32 v34, v122
	v_accvgpr_write_b32 a28, v0
	v_cvt_f16_f32_e32 v0, v36
	v_cvt_f16_f32_e32 v36, v119
	v_cvt_f16_f32_e32 v119, v225
	v_cvt_f16_f32_e32 v67, v124
	v_accvgpr_write_b32 a36, v0
	v_cvt_f16_f32_e32 v0, v38
	v_cvt_f16_f32_e32 v38, v118
	v_cvt_f16_f32_e32 v30, v126
	v_cvt_f16_f32_e32 v28, v127
	v_accvgpr_write_b32 a29, v0
	v_cvt_f16_f32_e32 v0, v39
	v_cvt_f16_f32_e32 v63, v132
	v_cvt_f16_f32_e32 v96, v133
	v_cvt_f16_f32_e32 v59, v140
	v_accvgpr_write_b32 a31, v0
	v_accvgpr_read_b32 v0, a152
	v_cvt_f16_f32_e32 v0, v0
	v_accvgpr_read_b32 v1, a153
	v_accvgpr_read_b32 v2, a154
	v_accvgpr_read_b32 v3, a155
	v_accvgpr_write_b32 a40, v0
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v98, v141
	v_cvt_f16_f32_e32 v55, v148
	v_cvt_f16_f32_e32 v100, v149
	v_accvgpr_write_b32 a41, v0
	v_cvt_f16_f32_e32 v0, v2
	v_cvt_f16_f32_e32 v51, v156
	v_cvt_f16_f32_e32 v102, v157
	v_cvt_f16_f32_e32 v47, v164
	v_accvgpr_write_b32 a33, v0
	v_cvt_f16_f32_e32 v0, v3
	v_cvt_f16_f32_e32 v104, v165
	v_cvt_f16_f32_e32 v43, v172
	v_cvt_f16_f32_e32 v106, v173
	v_accvgpr_write_b32 a35, v0
	v_accvgpr_read_b32 v0, a156
	v_cvt_f16_f32_e32 v0, v0
	v_accvgpr_read_b32 v2, a158
	v_accvgpr_read_b32 v3, a159
	v_accvgpr_read_b32 v1, a157
	v_accvgpr_write_b32 a42, v0
	v_cvt_f16_f32_e32 v0, v2
	v_cvt_f16_f32_e32 v20, v1
	v_cvt_f16_f32_e32 v39, v180
	v_cvt_f16_f32_e32 v108, v181
	v_accvgpr_write_b32 a38, v0
	v_cvt_f16_f32_e32 v0, v3
	v_cvt_f16_f32_e32 v35, v188
	v_cvt_f16_f32_e32 v110, v189
	v_cvt_f16_f32_e32 v31, v196
	v_accvgpr_write_b32 a39, v0
	v_accvgpr_read_b32 v0, a160
	v_cvt_f16_f32_e32 v0, v0
	v_accvgpr_read_b32 v2, a162
	v_accvgpr_read_b32 v3, a163
	v_accvgpr_read_b32 v1, a161
	v_accvgpr_write_b32 a43, v0
	v_cvt_f16_f32_e32 v0, v2
	v_cvt_f16_f32_e32 v18, v1
	v_cvt_f16_f32_e32 v112, v197
	v_cvt_f16_f32_e32 v114, v205
	v_accvgpr_write_b32 a45, v0
	v_cvt_f16_f32_e32 v0, v3
	v_cvt_f16_f32_e32 v23, v212
	v_cvt_f16_f32_e32 v116, v213
	v_cvt_f16_f32_e32 v19, v220
	v_accvgpr_write_b32 a46, v0
	v_accvgpr_read_b32 v0, a164
	v_cvt_f16_f32_e32 v0, v0
	v_accvgpr_read_b32 v2, a166
	v_accvgpr_read_b32 v3, a167
	v_accvgpr_read_b32 v1, a165
	v_accvgpr_write_b32 a44, v0
	v_cvt_f16_f32_e32 v0, v2
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v6, v7
	v_cvt_f16_f32_e32 v4, v1
	v_accvgpr_write_b32 a47, v0
	v_cvt_f16_f32_e32 v0, v3
	v_mov_b32_e32 v92, v6
	v_cvt_f16_f32_e32 v118, v221
	v_cvt_f16_f32_e32 v120, v229
	v_accvgpr_write_b32 a48, v0
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v122, v237
	v_cvt_f16_f32_e32 v7, v244
	v_cvt_f16_f32_e32 v124, v245
	v_accvgpr_write_b32 a50, v0
	v_cvt_f16_f32_e32 v0, v9
	v_accvgpr_read_b32 v8, a172
	v_accvgpr_read_b32 v10, a174
	v_accvgpr_read_b32 v11, a175
	v_accvgpr_write_b32 a51, v0
	v_cvt_f16_f32_e32 v0, v10
	v_accvgpr_read_b32 v9, a173
	v_cvt_f16_f32_e32 v70, v9
	v_cvt_f16_f32_e32 v9, v240
	v_accvgpr_write_b32 a54, v0
	v_cvt_f16_f32_e32 v0, v11
	v_accvgpr_read_b32 v10, a176
	v_accvgpr_read_b32 v12, a178
	v_accvgpr_read_b32 v13, a179
	v_accvgpr_write_b32 a56, v0
	v_cvt_f16_f32_e32 v0, v12
	v_cvt_f16_f32_e32 v12, v26
	v_cvt_f16_f32_e32 v72, v10
	v_cvt_f16_f32_e32 v10, v91
	v_accvgpr_write_b32 a62, v0
	v_cvt_f16_f32_e32 v0, v13
	v_cvt_f16_f32_e32 v91, v113
	v_cvt_f16_f32_e32 v113, v201
	v_cvt_f16_f32_e32 v13, v232
	v_accvgpr_write_b32 a63, v0
	v_cvt_f16_f32_e32 v0, v14
	v_accvgpr_read_b32 v6, a63
	v_cvt_f16_f32_e32 v14, v90
	v_cvt_f16_f32_e32 v90, v109
	v_accvgpr_write_b32 a61, v0
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v109, v185
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v29, v29, v113
	v_accvgpr_read_b32 v11, a177
	v_accvgpr_write_b32 a53, v0
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v0, v24
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v37, v37, v109
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v76, v11
	v_accvgpr_write_b32 a57, v0
	v_cvt_f16_f32_e32 v0, v25
	v_accvgpr_read_b32 v24, a188
	v_accvgpr_read_b32 v25, a189
	v_accvgpr_read_b32 v26, a190
	v_accvgpr_write_b32 a58, v0
	v_cvt_f16_f32_e32 v0, v24
	v_accvgpr_read_b32 v27, a191
	v_cvt_f16_f32_e32 v255, v26
	v_cvt_f16_f32_e32 v15, v228
	v_accvgpr_write_b32 a60, v0
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v25, v208
	v_cvt_f16_f32_e32 v11, v236
	v_cvt_f16_f32_e32 v3, v252
	v_accvgpr_write_b32 a59, v0
	v_cvt_f16_f32_e32 v0, v27
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v25, v25, v115
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v27, v204
	v_cvt_f16_f32_e32 v126, v253
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v87, v255, v0
	v_accvgpr_read_b32 v0, a62
	v_pack_b32_f16 v93, v0, v6
	v_accvgpr_read_b32 v0, a54
	v_accvgpr_read_b32 v6, a56
	v_pack_b32_f16 v95, v0, v6
	v_accvgpr_read_b32 v0, a50
	v_accvgpr_read_b32 v6, a51
	v_pack_b32_f16 v97, v0, v6
	v_accvgpr_read_b32 v0, a47
	v_accvgpr_read_b32 v6, a48
	v_pack_b32_f16 v99, v0, v6
	v_accvgpr_read_b32 v0, a45
	v_accvgpr_read_b32 v6, a46
	v_pack_b32_f16 v101, v0, v6
	v_accvgpr_read_b32 v0, a38
	v_accvgpr_read_b32 v6, a39
	v_pack_b32_f16 v103, v0, v6
	v_accvgpr_read_b32 v0, a33
	v_accvgpr_read_b32 v6, a35
	v_pack_b32_f16 v105, v0, v6
	v_accvgpr_read_b32 v0, a29
	v_accvgpr_read_b32 v6, a31
	v_pack_b32_f16 v107, v0, v6
	v_accvgpr_read_b32 v0, a26
	v_accvgpr_read_b32 v6, a28
	v_pack_b32_f16 v109, v0, v6
	v_accvgpr_read_b32 v0, a23
	v_accvgpr_read_b32 v6, a24
	v_pack_b32_f16 v111, v0, v6
	v_accvgpr_read_b32 v0, a18
	v_accvgpr_read_b32 v6, a19
	v_pack_b32_f16 v113, v0, v6
	v_accvgpr_read_b32 v0, a16
	v_accvgpr_read_b32 v6, a17
	v_pack_b32_f16 v115, v0, v6
	v_accvgpr_read_b32 v0, a13
	v_accvgpr_read_b32 v6, a14
	v_pack_b32_f16 v117, v0, v6
	v_accvgpr_read_b32 v0, a9
	v_accvgpr_read_b32 v6, a10
	v_accvgpr_read_b32 v193, a193
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v1, v194
	v_cvt_f16_f32_e32 v127, v195
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v17, v17, v119
	v_pack_b32_f16 v119, v0, v6
	v_accvgpr_read_b32 v0, a6
	v_accvgpr_read_b32 v6, a7
	.loc	1 92 9                          ; 00_naive_mfma.py:92:9
	v_cvt_f16_f32_e32 v26, v130
	v_cvt_f16_f32_e32 v24, v131
	v_cvt_f16_f32_e32 v128, v134
	v_cvt_f16_f32_e32 v130, v135
	v_cvt_f16_f32_e32 v129, v138
	v_cvt_f16_f32_e32 v131, v139
	v_cvt_f16_f32_e32 v138, v142
	v_cvt_f16_f32_e32 v135, v143
	v_cvt_f16_f32_e32 v136, v146
	v_cvt_f16_f32_e32 v137, v147
	v_cvt_f16_f32_e32 v133, v150
	v_cvt_f16_f32_e32 v132, v151
	v_cvt_f16_f32_e32 v139, v154
	v_cvt_f16_f32_e32 v134, v155
	v_cvt_f16_f32_e32 v154, v158
	v_cvt_f16_f32_e32 v155, v159
	v_cvt_f16_f32_e32 v153, v162
	v_cvt_f16_f32_e32 v157, v163
	v_cvt_f16_f32_e32 v148, v166
	v_cvt_f16_f32_e32 v147, v170
	v_cvt_f16_f32_e32 v151, v171
	v_cvt_f16_f32_e32 v146, v174
	v_cvt_f16_f32_e32 v150, v175
	v_cvt_f16_f32_e32 v149, v178
	v_cvt_f16_f32_e32 v174, v179
	v_cvt_f16_f32_e32 v144, v182
	v_cvt_f16_f32_e32 v172, v183
	v_cvt_f16_f32_e32 v143, v186
	v_cvt_f16_f32_e32 v173, v187
	v_cvt_f16_f32_e32 v142, v190
	v_cvt_f16_f32_e32 v168, v191
	v_cvt_f16_f32_e32 v141, v198
	v_cvt_f16_f32_e32 v166, v199
	v_cvt_f16_f32_e32 v140, v202
	v_cvt_f16_f32_e32 v169, v203
	v_cvt_f16_f32_e32 v187, v206
	v_cvt_f16_f32_e32 v164, v207
	v_cvt_f16_f32_e32 v191, v210
	v_cvt_f16_f32_e32 v163, v211
	v_cvt_f16_f32_e32 v186, v214
	v_cvt_f16_f32_e32 v162, v215
	v_cvt_f16_f32_e32 v188, v218
	v_cvt_f16_f32_e32 v165, v219
	v_cvt_f16_f32_e32 v184, v222
	v_cvt_f16_f32_e32 v161, v223
	v_cvt_f16_f32_e32 v183, v226
	v_cvt_f16_f32_e32 v160, v227
	v_cvt_f16_f32_e32 v185, v230
	v_cvt_f16_f32_e32 v159, v231
	v_cvt_f16_f32_e32 v180, v234
	v_cvt_f16_f32_e32 v156, v235
	v_cvt_f16_f32_e32 v179, v238
	v_cvt_f16_f32_e32 v158, v239
	v_cvt_f16_f32_e32 v178, v242
	v_cvt_f16_f32_e32 v182, v243
	v_cvt_f16_f32_e32 v181, v246
	v_cvt_f16_f32_e32 v177, v247
	v_cvt_f16_f32_e32 v176, v250
	v_cvt_f16_f32_e32 v175, v251
	v_cvt_f16_f32_e32 v171, v192
	v_cvt_f16_f32_e32 v170, v193
	.loc	1 68 11                         ; 00_naive_mfma.py:68:11
	v_pack_b32_f16 v13, v13, v121
	v_pack_b32_f16 v121, v0, v6
	v_accvgpr_read_b32 v0, a4
	v_accvgpr_read_b32 v6, a5
	v_pack_b32_f16 v9, v9, v123
	v_pack_b32_f16 v123, v0, v6
	v_accvgpr_read_b32 v0, a2
	v_accvgpr_read_b32 v6, a3
	v_pack_b32_f16 v5, v5, v125
	v_pack_b32_f16 v125, v0, v6
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v6, a1
	v_pack_b32_f16 v1, v1, v127
	v_pack_b32_f16 v3, v3, v126
	v_pack_b32_f16 v7, v7, v124
	v_pack_b32_f16 v11, v11, v122
	v_pack_b32_f16 v15, v15, v120
	v_pack_b32_f16 v19, v19, v118
	v_pack_b32_f16 v23, v23, v116
	v_pack_b32_f16 v27, v27, v114
	v_pack_b32_f16 v31, v31, v112
	v_accvgpr_read_b32 v112, a55
	v_pack_b32_f16 v35, v35, v110
	v_mov_b32_e32 v110, v74
	v_pack_b32_f16 v39, v39, v108
	v_mov_b32_e32 v108, v68
	v_pack_b32_f16 v43, v43, v106
	v_mov_b32_e32 v106, v66
	v_pack_b32_f16 v47, v47, v104
	v_mov_b32_e32 v104, v64
	v_pack_b32_f16 v51, v51, v102
	v_mov_b32_e32 v102, v22
	v_accvgpr_read_b32 v22, a52
	v_pack_b32_f16 v55, v55, v100
	v_mov_b32_e32 v100, v20
	v_accvgpr_read_b32 v20, a49
	v_pack_b32_f16 v59, v59, v98
	v_mov_b32_e32 v98, v18
	v_pack_b32_f16 v63, v63, v96
	v_mov_b32_e32 v96, v4
	v_accvgpr_read_b32 v4, a37
	v_pack_b32_f16 v67, v67, v94
	v_mov_b32_e32 v94, v2
	v_accvgpr_read_b32 v2, a34
	v_pack_b32_f16 v73, v73, v91
	v_pack_b32_f16 v75, v75, v90
	v_mov_b32_e32 v90, v8
	v_accvgpr_read_b32 v8, a30
	v_pack_b32_f16 v77, v77, v89
	v_pack_b32_f16 v79, v79, v88
	v_mov_b32_e32 v88, v70
	v_pack_b32_f16 v83, v83, v86
	v_mov_b32_e32 v86, v72
	v_pack_b32_f16 v85, v84, v85
	v_mov_b32_e32 v84, v76
	v_pack_b32_f16 v89, v12, v254
	v_accvgpr_read_b32 v12, a22
	v_pack_b32_f16 v91, v82, v80
	v_accvgpr_read_b32 v199, a61
	v_accvgpr_read_b32 v190, a53
	v_accvgpr_read_b32 v192, a57
	v_accvgpr_read_b32 v195, a44
	v_accvgpr_read_b32 v194, a58
	v_accvgpr_read_b32 v197, a40
	v_accvgpr_read_b32 v196, a43
	v_accvgpr_read_b32 v193, a60
	v_accvgpr_read_b32 v201, a41
	v_accvgpr_read_b32 v72, a59
	v_accvgpr_read_b32 v82, a21
	v_accvgpr_read_b32 v80, a25
	v_accvgpr_read_b32 v202, a32
	v_accvgpr_read_b32 v200, a36
	v_accvgpr_read_b32 v198, a42
	v_mov_b32_e32 v70, v78
	v_accvgpr_read_b32 v78, a15
	v_accvgpr_read_b32 v76, a20
	v_accvgpr_read_b32 v74, a27
	v_mov_b32_e32 v68, v16
	v_accvgpr_read_b32 v16, a11
	v_mov_b32_e32 v66, v14
	v_accvgpr_read_b32 v14, a12
	v_mov_b32_e32 v64, v10
	v_accvgpr_read_b32 v10, a8
	v_pack_b32_f16 v127, v0, v6
	v_accvgpr_read_b32 v0, a108
	v_accvgpr_read_b32 v18, a109
	s_branch .LBB0_5
.LBB0_4:
	.loc	1 0 11 is_stmt 0                ; 00_naive_mfma.py:0:11
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v127, 0
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
.LBB0_5:                                ; %._crit_edge
	.loc	1 96 15 is_stmt 1               ; 00_naive_mfma.py:96:15
	v_and_b32_e32 v0, 15, v0
	s_lshr_b32 s0, s15, 3
	v_and_or_b32 v0, s0, 16, v0
	.loc	1 98 22                         ; 00_naive_mfma.py:98:22
	s_mul_i32 s0, s16, s13
	.loc	1 98 14 is_stmt 0               ; 00_naive_mfma.py:98:14
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	.loc	1 103 5 is_stmt 1               ; 00_naive_mfma.py:103:5
	s_mov_b32 s33, 0x5040100
	.loc	1 98 14                         ; 00_naive_mfma.py:98:14
	s_addc_u32 s3, s7, s1
	s_ashr_i32 s15, s14, 31
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v126, v2, v4, s33
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_lshrrev_b32_e32 v2, 2, v18
	.loc	1 98 14                         ; 00_naive_mfma.py:98:14
	s_lshl_b64 s[0:1], s[14:15], 1
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_and_b32_e32 v2, 28, v2
	.loc	1 98 14                         ; 00_naive_mfma.py:98:14
	s_add_u32 s36, s2, s0
	s_addc_u32 s0, s3, s1
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[28:29], s8, v0
	.loc	1 101 40 is_stmt 0              ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e32 vcc, s9, v2
	.loc	1 99 17 is_stmt 1               ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v4, v0, s13
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	s_and_b32 s37, s0, 0xffff
	v_add_lshl_u32 v6, v4, v2, 1
	v_bfrev_b32_e32 v189, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[0:1], s[28:29], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v6, v189, v6, s[0:1]
	buffer_store_dwordx2 v[126:127], v6, s[36:39], 0 offen
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_or_b32_e32 v6, 32, v2
	.loc	1 101 40                        ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e64 s[0:1], s9, v6
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v124, v8, v112, s33
	v_add_lshl_u32 v8, v6, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[2:3], s[28:29], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_cndmask_b32_e64 v8, v189, v8, s[2:3]
	buffer_store_dwordx2 v[124:125], v8, s[36:39], 0 offen
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_or_b32_e32 v8, 64, v2
	.loc	1 101 40                        ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e64 s[2:3], s9, v8
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v122, v10, v12, s33
	v_add_lshl_u32 v10, v8, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[4:5], s[28:29], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_cndmask_b32_e64 v10, v189, v10, s[4:5]
	buffer_store_dwordx2 v[122:123], v10, s[36:39], 0 offen
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_or_b32_e32 v122, 0x60, v2
	.loc	1 101 40                        ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e64 s[4:5], s9, v122
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v120, v14, v16, s33
	v_add_lshl_u32 v14, v122, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[6:7], s[28:29], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_cndmask_b32_e64 v14, v189, v14, s[6:7]
	buffer_store_dwordx2 v[120:121], v14, s[36:39], 0 offen
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_or_b32_e32 v120, 0x80, v2
	.loc	1 101 40                        ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e64 s[6:7], s9, v120
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v118, v20, v78, s33
	v_add_lshl_u32 v20, v120, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[10:11], s[28:29], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_cndmask_b32_e64 v20, v189, v20, s[10:11]
	buffer_store_dwordx2 v[118:119], v20, s[36:39], 0 offen
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_or_b32_e32 v118, 0xa0, v2
	.loc	1 96 15                         ; 00_naive_mfma.py:96:15
	v_or_b32_e32 v10, 32, v0
	v_or_b32_e32 v12, 64, v0
	v_or_b32_e32 v14, 0x60, v0
	v_or_b32_e32 v16, 0x80, v0
	v_or_b32_e32 v18, 0xa0, v0
	v_or_b32_e32 v20, 0xc0, v0
	v_or_b32_e32 v0, 0xe0, v0
	.loc	1 97 15                         ; 00_naive_mfma.py:97:15
	v_or_b32_e32 v119, 0xc0, v2
	v_or_b32_e32 v121, 0xe0, v2
	.loc	1 101 40                        ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e64 s[14:15], s9, v118
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v116, v22, v110, s33
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v22, v10, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[30:31], s8, v10
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v10, v12, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[24:25], s8, v12
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v12, v14, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[22:23], s8, v14
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v14, v16, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[20:21], s8, v16
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v16, v18, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[18:19], s8, v18
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v123, v20, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[16:17], s8, v20
	.loc	1 99 17                         ; 00_naive_mfma.py:99:17
	v_mul_lo_u32 v124, v0, s13
	.loc	1 101 15                        ; 00_naive_mfma.py:101:15
	v_cmp_gt_i32_e64 s[26:27], s8, v0
	.loc	1 101 40 is_stmt 0              ; 00_naive_mfma.py:101:40
	v_cmp_gt_i32_e64 s[12:13], s9, v119
	v_cmp_gt_i32_e64 s[10:11], s9, v121
	.loc	1 103 5 is_stmt 1               ; 00_naive_mfma.py:103:5
	v_add_lshl_u32 v0, v118, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[28:29], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[116:117], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v119, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[28:29], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v114, v76, v82, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[114:115], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v121, v4, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[28:29], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v112, v80, v108, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[112:113], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v110, v106, v74, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[110:111], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v108, v104, v202, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[108:109], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v106, v102, v200, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[106:107], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v104, v201, v197, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[104:105], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v102, v100, v198, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[102:103], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v100, v98, v196, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[100:101], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v98, v96, v195, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[98:99], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v22, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[30:31], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v96, v92, v94, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[96:97], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v94, v88, v90, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[94:95], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v92, v84, v86, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[92:93], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v90, v190, v199, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[90:91], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v88, v194, v192, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[88:89], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v86, v72, v193, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[86:87], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v84, v68, v70, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[84:85], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v82, v64, v66, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[82:83], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v10, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[24:25], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v80, v60, v62, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[80:81], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v78, v56, v58, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[78:79], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v76, v52, v54, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[76:77], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v74, v48, v50, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[74:75], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v72, v44, v46, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[72:73], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v70, v40, v42, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[70:71], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v68, v36, v38, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[68:69], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v66, v32, v34, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[66:67], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v12, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[22:23], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v64, v28, v30, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[64:65], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v62, v24, v26, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[62:63], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v60, v130, v128, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[60:61], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v58, v131, v129, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[58:59], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v56, v135, v138, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[56:57], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v54, v137, v136, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[54:55], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v52, v132, v133, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[52:53], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v50, v134, v139, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[50:51], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v14, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[20:21], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v48, v155, v154, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[48:49], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v46, v157, v153, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[46:47], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v44, v152, v148, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[44:45], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v42, v151, v147, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[42:43], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v40, v150, v146, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[40:41], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v38, v174, v149, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[38:39], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v36, v172, v144, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[36:37], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v34, v173, v143, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[34:35], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v16, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[18:19], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v32, v168, v142, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[32:33], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v30, v167, v145, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[30:31], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v28, v166, v141, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[28:29], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v26, v169, v140, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[26:27], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v24, v164, v187, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[24:25], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v22, v163, v191, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[22:23], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v20, v162, v186, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[20:21], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v18, v165, v188, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[18:19], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v123, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 s[8:9], s[16:17], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v16, v161, v184, s33
	v_cndmask_b32_e64 v0, v189, v0, s[8:9]
	buffer_store_dwordx2 v[16:17], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v2, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], vcc
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v14, v160, v183, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[14:15], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v6, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[0:1]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v12, v159, v185, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[12:13], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v8, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[2:3]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v10, v156, v180, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[10:11], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v122, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[4:5]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v8, v158, v179, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[8:9], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v120, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[6:7]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v6, v182, v178, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[6:7], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v118, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[14:15]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v4, v177, v181, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[4:5], v0, s[36:39], 0 offen
	v_add_lshl_u32 v0, v124, v119, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[12:13]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v2, v175, v176, s33
	v_cndmask_b32_e32 v0, v189, v0, vcc
	buffer_store_dwordx2 v[2:3], v0, s[36:39], 0 offen
	v_add_lshl_u32 v2, v124, v121, 1
	.loc	1 101 14                        ; 00_naive_mfma.py:101:14
	s_and_b64 vcc, s[26:27], s[10:11]
	.loc	1 103 5                         ; 00_naive_mfma.py:103:5
	v_perm_b32 v0, v170, v171, s33
	v_cndmask_b32_e32 v2, v189, v2, vcc
	buffer_store_dwordx2 v[0:1], v2, s[36:39], 0 offen
	.loc	1 8 1                           ; 00_naive_mfma.py:8:1
	s_endpgm
.Ltmp7:
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
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 40
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
	.set .Lv0_naive.num_agpr, 256
	.set .Lv0_naive.numbered_sgpr, 40
	.set .Lv0_naive.num_named_barrier, 0
	.set .Lv0_naive.private_seg_size, 0
	.set .Lv0_naive.uses_vcc, 1
	.set .Lv0_naive.uses_flat_scratch, 0
	.set .Lv0_naive.has_dyn_sized_stack, 0
	.set .Lv0_naive.has_recursion, 0
	.set .Lv0_naive.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13748
; TotalNumSgprs: 46
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 512
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
  - .agpr_count:     256
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
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         v0_naive.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
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
