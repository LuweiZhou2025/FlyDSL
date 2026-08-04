# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# fp8 GEMM (C = B * A, 输出 bf16) —— 8-wave 版本，按 test_gemm_v9_fp8.py 的 tile + layout
# 抽象风格编写（flat_divide / make_tiled_copy / make_tiled_mma / make_fragment / fx.copy /
# fx.gemm），不做手动 byte-offset DMA。算法对标 test_gemm.py::compile_gemm_950 的
# gemm_8wave_950（fp8）：2x2 quadrant、8 wave（tiled_mma wave grid 4x2）、双缓冲 LDS、
# 每 region compute-phase(s_setprio + s_barrier) 调度。
#   - BLOCK_M=BLOCK_N=BLOCK_K=128, TILE_M=TILE_N=256, block=512(8 wave)
#   - MFMA V_MFMA_SCALE_F32_16X16X128_F8F6F4（scale=0）
#   - A/B LDS dual-padding（[[1024,16],[2048,32]]）消 bank conflict；tile-based fx.copy g2s。
#   - 约定：A 走 make_fragment_B，B 走 make_fragment_A；fx.gemm(mma, C, frag_B, frag_A)。
#
# 运行：cd /mywork/FlyDSL/tests/kernels && HIP_VISIBLE_DEVICES=4 python ./test_gemm_v9_fp8_8wave.py

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float32, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector, arith
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def div_up(x, y):
    return (x + y - 1) // y


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def compile_gemm_fp8_8wave(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    pid_swizzle=True,
    permlane_epilogue=True,
    preshuffle_b=False,
    with_scale=False,
):
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    element_type = fx.Float8E4M3FN
    elements_per_128b = 16  # 128bit / fp8(8bit)

    def _get_pids_950(pid, M, GRID_MN, NUM_XCDS, GROUP_SIZE_M):
        num_pid_m = (M + TILE_M - 1) // TILE_M
        num_pid_n = div_up(N, TILE_N)
        if const_expr(NUM_XCDS != 1):
            pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
            tall_xcds = GRID_MN % NUM_XCDS
            tall_xcds = (tall_xcds == 0).select(NUM_XCDS, tall_xcds)
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            if xcd < tall_xcds:
                pid = xcd * pids_per_xcd + local_pid
            else:
                pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
        if const_expr(GROUP_SIZE_M == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            remaining_pid_m = num_pid_m - first_pid_m
            group_size_m = (remaining_pid_m < GROUP_SIZE_M).select(remaining_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # A/B LDS dual padding（对标 gemm_4wave_950 fp8：[[1024,16],[2048,32]]）
    A_GROUP = 8 * BLOCK_K + 16
    a_lds_elems = 2 * A_GROUP + 32  # 每 2 组再 pad 32
    a_lds_elems = (BLOCK_M // 16) * a_lds_elems  # 8 * (2*(8*128+16)+32) = 8*2112 = 16896

    @fx.struct
    class LDS:
        a_t0: fx.Array[Float8E4M3FN, 16896, 16]
        a_t1: fx.Array[Float8E4M3FN, 16896, 16]
        a_b0: fx.Array[Float8E4M3FN, 16896, 16]
        a_b1: fx.Array[Float8E4M3FN, 16896, 16]
        b_l0: fx.Array[Float8E4M3FN, 16896, 16]
        b_l1: fx.Array[Float8E4M3FN, 16896, 16]
        b_r0: fx.Array[Float8E4M3FN, 16896, 16]
        b_r1: fx.Array[Float8E4M3FN, 16896, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def gemm_kernel(argA: fx.Tensor, argB: fx.Tensor, argC: fx.Tensor,
                    argScaleA: fx.Tensor, argScaleB: fx.Tensor, M: int):
        tid = fx.thread_idx.x
        wave_id = tid // 64
        num_pid_n = div_up(N, TILE_N)
        if const_expr(pid_swizzle):
            bid_x, bid_y = get_pids_950(fx.block_idx.x, M, fx.grid_dim.x, 8, 4)
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n

        a_iter = fx.recast_iter(element_type, fx.get_iter(argA))
        b_iter = fx.recast_iter(element_type, fx.get_iter(argB))
        A_2d = fx.Tensor(fx.make_view(a_iter, fx.make_layout((M, K), (K, 1))))
        B_2d = fx.Tensor(fx.make_view(b_iter, fx.make_layout((N, K), (K, 1))))
        C_2d = fx.Tensor(fx.make_view(fx.get_iter(argC), fx.make_layout((M, N), (N, 1))))

        A = fx.rocdl.make_buffer_tensor(A_2d, max_size=False)
        B = fx.rocdl.make_buffer_tensor(B_2d, max_size=False)
        C = fx.rocdl.make_buffer_tensor(C_2d, max_size=False)
        c_store_rsrc = fx.buffer_ops.create_buffer_resource(argC, max_size=True)

        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 0, None]
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 1, None]
        bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 0, None]
        bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 1, None]
        # 分组全局视图（每 8 行为一组，映射到 padding LDS 行组）
        a_grouped = fx.make_layout(
            ((8, BLOCK_M // 8), BLOCK_K, K // BLOCK_K),
            ((BLOCK_M // 8 * K, K), 1, BLOCK_K),
        )
        bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), a_grouped))
        bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), a_grouped))
        b_grouped = fx.make_layout(
            ((8, BLOCK_N // 8), BLOCK_K, K // BLOCK_K),
            ((BLOCK_N // 8 * K, K), 1, BLOCK_K),
        )
        bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), b_grouped))
        bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), b_grouped))

        # preshuffle B：host 端 shuffle_weight(B, layout=(16,64))，kernel 用 subB 再视图
        if const_expr(preshuffle_b):
            _subB = fx.make_layout(
                ((16, BLOCK_N // 16), (16, BLOCK_K // 16), K // BLOCK_K),
                ((16, 16 * K), (1, 256), 2048),
            )
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), _subB))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), _subB))

        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 0]
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 1]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 0]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 1]

        # ---- tiled MMA: 8 wave (wave grid 4x2)，fp8 A/B swap => (4,2,1),(1,4,0) ----
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, element_type))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((4, 2, 1), (1, 4, 0)), (None, None, k_perm))
        thr_mma = tiled_mma.thr_slice(tid)

        async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)

        lds = fx.SharedAllocator().allocate(LDS).peek()

        # A/B LDS dual padding write/read layout（对标 gemm_4wave_950 fp8）
        _wr = fx.make_layout(
            ((8, 2, BLOCK_M // 16), BLOCK_K),
            ((BLOCK_K, 8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32), 1),
        )
        _rd = fx.make_layout(
            ((2, BLOCK_M // 16, 8), (32, BLOCK_K // 32)),
            ((8 * BLOCK_K + 16, 2 * (8 * BLOCK_K + 16) + 32, BLOCK_K), (1, 32)),
        )
        # 8 wave g2s DMA tv：512 线程，tile(64, BLOCK_K)，每线程 2 次 128-bit load。
        _a_dma_tv = fx.make_layout(
            ((8, 8, 8), elements_per_128b),
            ((elements_per_128b * 64, 1, 8), 64),
        )
        dma = fx.make_tiled_copy(buffer_copy_atom, _a_dma_tv, fx.make_tile(64, BLOCK_K)).get_slice(tid)

        # B LDS wr/rd：preshuffle 时用与 shuffle 一致的无 bank-conflict 布局（wr==rd），
        # 否则沿用与 A 相同的 dual-padding _wr/_rd。
        _wr_b = _wr
        _rd_b = _rd
        if const_expr(preshuffle_b):
            _b_lds = fx.make_layout(((16, BLOCK_N // 16), (16, BLOCK_K // 16)), ((16, 2048), (1, 256)))
            _wr_b = _b_lds
            _rd_b = _b_lds
            # B 专属 g2s DMA（512 线程，tile(64,BLOCK_K)）：对标 4-wave 的 ((16,8,2),16),((1,512,16),32)
            # tile(32)，8-wave 行数翻倍 => (16,8,4),(1,1024,16),64 tile(64)。
            _b_g2s_tv = fx.make_layout(((16, 8, 4), elements_per_128b), ((1, 1024, 16), 64))
            dma_b = fx.make_tiled_copy(buffer_copy_atom, _b_g2s_tv, fx.make_tile(64, BLOCK_K)).get_slice(tid)
        else:
            dma_b = dma

        sA_t_wr = [fx.make_view(lds.a_t0.ptr, _wr), fx.make_view(lds.a_t1.ptr, _wr)]
        sA_b_wr = [fx.make_view(lds.a_b0.ptr, _wr), fx.make_view(lds.a_b1.ptr, _wr)]
        sA_t_rd = [fx.make_view(lds.a_t0.ptr, _rd), fx.make_view(lds.a_t1.ptr, _rd)]
        sA_b_rd = [fx.make_view(lds.a_b0.ptr, _rd), fx.make_view(lds.a_b1.ptr, _rd)]
        sB_l_wr = [fx.make_view(lds.b_l0.ptr, _wr_b), fx.make_view(lds.b_l1.ptr, _wr_b)]
        sB_r_wr = [fx.make_view(lds.b_r0.ptr, _wr_b), fx.make_view(lds.b_r1.ptr, _wr_b)]
        sB_l_rd = [fx.make_view(lds.b_l0.ptr, _rd_b), fx.make_view(lds.b_l1.ptr, _rd_b)]
        sB_r_rd = [fx.make_view(lds.b_r0.ptr, _rd_b), fx.make_view(lds.b_r1.ptr, _rd_b)]

        aT_g = dma.partition_S(bA_t)
        aB_g = dma.partition_S(bA_b)
        bL_g = dma_b.partition_S(bB_l)
        bR_g = dma_b.partition_S(bB_r)
        aT_s = [dma.partition_D(sA_t_wr[0]), dma.partition_D(sA_t_wr[1])]
        aB_s = [dma.partition_D(sA_b_wr[0]), dma.partition_D(sA_b_wr[1])]
        bL_s = [dma_b.partition_D(sB_l_wr[0]), dma_b.partition_D(sB_l_wr[1])]
        bR_s = [dma_b.partition_D(sB_r_wr[0]), dma_b.partition_D(sB_r_wr[1])]

        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        copy_b = fx.make_tiled_copy_A(lds_copy_atom, tiled_mma).get_slice(tid)
        s2r_src0_A_t = copy_a.partition_S(sA_t_rd[0])
        s2r_src0_A_b = copy_a.partition_S(sA_b_rd[0])
        s2r_src0_B_l = copy_b.partition_S(sB_l_rd[0])
        s2r_src0_B_r = copy_b.partition_S(sB_r_rd[0])
        s2r_src1_A_t = copy_a.partition_S(sA_t_rd[1])
        s2r_src1_A_b = copy_a.partition_S(sA_b_rd[1])
        s2r_src1_B_l = copy_b.partition_S(sB_l_rd[1])
        s2r_src1_B_r = copy_b.partition_S(sB_r_rd[1])

        frag_A_t = thr_mma.make_fragment_B(sA_t_rd[0])
        frag_B_l = thr_mma.make_fragment_A(sB_l_rd[0])
        frag_B_r = thr_mma.make_fragment_A(sB_r_rd[0])
        dest_frag_A_t = copy_a.retile(frag_A_t)
        dest_frag_B_l = copy_b.retile(frag_B_l)
        dest_frag_B_r = copy_b.retile(frag_B_r)

        # ---- C fragments：转置 tile + make_fragment_C（对标 gemm_8wave_950，无 select）----
        transposed_c_layout = fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0))
        bC_tl = fx.composition(bC_tl, transposed_c_layout)
        bC_tr = fx.composition(bC_tr, transposed_c_layout)
        bC_bl = fx.composition(bC_bl, transposed_c_layout)
        bC_br = fx.composition(bC_br, transposed_c_layout)
        frag_C_tl = thr_mma.make_fragment_C(bC_tl)
        frag_C_tr = thr_mma.make_fragment_C(bC_tr)
        frag_C_bl = thr_mma.make_fragment_C(bC_bl)
        frag_C_br = thr_mma.make_fragment_C(bC_br)


        # ==== block-scale (a8w8) 设置：A per-token group-128，B block-wise 128x128 ====
        # C[m,n] = sum_kb scaleA[m,kb] * scaleB[n//128,kb] * (fp8 partial over k-block kb)。
        # 每个 k-tile(=BLOCK_K=128) 恰好是一个 scale block；对每象限先算无 scale partial
        # (frag_P)，再按 (scaleA_per_m ⊙ scaleB_scalar) FMA 累加进 frag_C。
        # C fragment 布局 [val=N, n0(N_REP), m0(M_REP)]；M 行 = quadrant_m*128 + m0*32
        #   + wave_m*16 + lane%16（wave_m=wave_id//4）=> scaleA 随 m0/lane 变化，广播 val/n0。
        M_REP = TILE_M // 64
        N_REP = TILE_N // 128
        if const_expr(with_scale):
            KB = K // 128
            sA_rsrc = fx.buffer_ops.create_buffer_resource(argScaleA, max_size=True)
            sB_rsrc = fx.buffer_ops.create_buffer_resource(argScaleB, max_size=True)
            lane_id = tid % 64
            wave_m = wave_id // 4
            sA_top_baseM = bid_x * TILE_M + wave_m * 16 + lane_id % 16
            sA_bot_baseM = sA_top_baseM + TILE_M // 2
            nb_l = bid_y * (TILE_N // 128)
            nb_r = nb_l + 1
            frag_P = thr_mma.make_fragment_C(bC_tl)

            def _load_sA(baseM, kb):
                base_off = baseM * KB + kb
                return [
                    fx.Float32(fx.buffer_ops.buffer_load(sA_rsrc, base_off + m0 * 32 * KB, vec_width=1))
                    for m0 in range_constexpr(M_REP)
                ]

            def _load_sB(nb, kb):
                return fx.Float32(fx.buffer_ops.buffer_load(sB_rsrc, nb * KB + kb, vec_width=1))

        def do_gemm(frag_C, frag_B, frag_A, sA_list, sB):
            if const_expr(with_scale):
                # 每 k-block 先算无 scale partial(frag_P)，再按 scaleA[m0]*scaleB FMA 累加进 frag_C。
                frag_P.fill(0)
                fx.gemm(mma_atom, frag_P, frag_B, frag_A, frag_P)
                for m0 in range_constexpr(M_REP):
                    s = sA_list[m0] * sB
                    for n0 in range_constexpr(N_REP):
                        cs = frag_C[None, n0, m0]
                        cs.store(cs.load() + frag_P[None, n0, m0].load() * s)
            else:
                fx.gemm(mma_atom, frag_C, frag_B, frag_A, frag_C)

        num_tiles = K // BLOCK_K
        assert num_tiles >= 4 and num_tiles % 2 == 0
        a_dsrd = frag_A_t.load().numel * element_type.width // 8 // 16
        b_dsrd = frag_B_l.load().numel * element_type.width // 8 // 16
        a_vmem = (BLOCK_M * BLOCK_K * element_type.width // 8) // (512 * 16)
        b_vmem = (BLOCK_N * BLOCK_K * element_type.width // 8) // (512 * 16)
        prologue_vmcnt = 3 * a_vmem + 3 * b_vmem
        ab_br_vmcnt = 2 * a_vmem + 3 * b_vmem
        bl_at_vmcnt = 3 * a_vmem + 2 * b_vmem

        def hot_loop_scheduler(dsrd_count, vmem_count):
            schedule_steps = max(dsrd_count, vmem_count)
            prev_dsrd = 0
            prev_vmem = 0
            for i in range_constexpr(schedule_steps):
                cur_dsrd = ((i + 1) * dsrd_count + schedule_steps - 1) // schedule_steps
                cur_vmem = ((i + 1) * vmem_count + schedule_steps - 1) // schedule_steps
                if const_expr(cur_dsrd > prev_dsrd):
                    rocdl.sched_dsrd(cur_dsrd - prev_dsrd)
                if const_expr(cur_vmem > prev_vmem):
                    rocdl.sched_vmem(cur_vmem - prev_vmem)
                prev_dsrd = cur_dsrd
                prev_vmem = cur_vmem
            rocdl.sched_barrier(0)

        def begin_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)

        def end_compute_phase():
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def wait_vmem_barrier(vmcnt):
            rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt, lgkmcnt=0))
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # ---- prologue：预取 tile0/tile1 到 LDS buf0/buf1，再 s2r buf0 的 A_t/B_l ----
        def do_g2s(kk, buf):
            ki = fx.Int32(kk)
            fx.copy(async_copy_atom, bL_g[None, None, None, ki], bL_s[buf])
            rocdl.sched_barrier(0)
            fx.copy(async_copy_atom, aT_g[None, None, None, ki], aT_s[buf])
            rocdl.sched_barrier(0)
            fx.copy(async_copy_atom, aB_g[None, None, None, ki], aB_s[buf])
            rocdl.sched_barrier(0)

            fx.copy(async_copy_atom, bR_g[None, None, None, ki], bR_s[buf])
            rocdl.sched_barrier(0)


        # ---- 非 scale 版：对标 pyhip gemm_8wave 的 4-phase 精确流水 ----
        # 每 tile 分 4 个 compute-phase（TL/TR/BL/BR，各一条 MFMA），phase 间穿插一次
        # ds_read + 一条 g2s 预取（读后即刷 LDS，barrier 保证全 wave 读完再覆盖）。
        # vmcnt 用精确值（a_vmem + 2*b_vmem）而非全 drain，让 g2s 与 MFMA 重叠。
        NS_VMCNT = a_vmem + 2 * b_vmem
        _lgkm0 = encode_waitcnt_950(lgkmcnt=0)
        _s2r_At = [s2r_src0_A_t, s2r_src1_A_t]
        _s2r_Ab = [s2r_src0_A_b, s2r_src1_A_b]
        _s2r_Bl = [s2r_src0_B_l, s2r_src1_B_l]
        _s2r_Br = [s2r_src0_B_r, s2r_src1_B_r]

        def _rd_At(b):
            fx.copy(lds_copy_atom, _s2r_At[b], dest_frag_A_t, pred=None)

        def _rd_Ab(b):
            fx.copy(lds_copy_atom, _s2r_Ab[b], dest_frag_A_t, pred=None)

        def _rd_Bl(b):
            fx.copy(lds_copy_atom, _s2r_Bl[b], dest_frag_B_l, pred=None)

        def _rd_Br(b):
            fx.copy(lds_copy_atom, _s2r_Br[b], dest_frag_B_r, pred=None)

        def _ld_At(b, ki):
            fx.copy(async_copy_atom, aT_g[None, None, None, ki], aT_s[b])

        def _ld_Ab(b, ki):
            fx.copy(async_copy_atom, aB_g[None, None, None, ki], aB_s[b])

        def _ld_Bl(b, ki):
            fx.copy(async_copy_atom, bL_g[None, None, None, ki], bL_s[b])

        def _ld_Br(b, ki):
            fx.copy(async_copy_atom, bR_g[None, None, None, ki], bR_s[b])

  
        rocdl.sched_barrier(0)
        _ld_Bl(0, 0)
        rocdl.sched_barrier(0)
        _ld_At(0, 0)
        rocdl.sched_barrier(0)
        _ld_Br(0, 0)
        rocdl.sched_barrier(0)
        _ld_Ab(0, 0)
        rocdl.sched_barrier(0)
        if wave_id >= 4:
            rocdl.s_barrier()
        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)
    
        vm_load_cnt_a = 2
        vm_load_cnt_b = 2
    
        rocdl.sched_barrier(0)
        vmcnt = vm_load_cnt_a + vm_load_cnt_b
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        _ld_At(1, 1)
        rocdl.sched_barrier(0)
        _ld_Bl(1, 1)
        rocdl.sched_barrier(0)
        _ld_Br(1, 1)
        rocdl.sched_barrier(0)
        
        vmcnt = vm_load_cnt_a + vm_load_cnt_b*2
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))

        acc_init = [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
        for kidx, states in range(0, num_tiles, 2, init=acc_init):
            frag_C_tl.store(states[0])
            frag_C_tr.store(states[1])
            frag_C_bl.store(states[2])
            frag_C_br.store(states[3])
            kiter = fx.Int32(kidx)

            if const_expr(not with_scale):
                tick = 0
                tock = 1
                _rd_Bl(tick)
                _rd_At(tick)
                _ld_Ab(tock, kiter+1)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
                rocdl.sched_barrier(0)
        
                begin_compute_phase()
                do_gemm(frag_C_tl, frag_B_l, frag_A_t, None, None)
                end_compute_phase()
                
                _rd_Br(tick)
                _ld_At(tick, kiter+2)

                begin_compute_phase()
                do_gemm(frag_C_tr, frag_B_r, frag_A_t, None, None)
                end_compute_phase()

                _rd_Ab(tick)
                _ld_Bl(tick, kiter+2)

                begin_compute_phase()
                do_gemm(frag_C_bl, frag_B_l, frag_A_t, None, None)
                end_compute_phase()

                _ld_Br(tick, kiter+2)
                rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vm_load_cnt_a + vm_load_cnt_b*2))
                
                begin_compute_phase()
                do_gemm(frag_C_br, frag_B_r, frag_A_t, None, None)
                end_compute_phase()
                
                
                tick = 1
                tock = 0
                _rd_Bl(tick)
                _rd_At(tick)
                _ld_Ab(tock, kiter+2)
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
                rocdl.sched_barrier(0)
        
                begin_compute_phase()
                do_gemm(frag_C_tl, frag_B_l, frag_A_t, None, None)
                end_compute_phase()
                
                _rd_Br(tick)
                _ld_At(tick, kiter+3)

                begin_compute_phase()
                do_gemm(frag_C_tr, frag_B_r, frag_A_t, None, None)
                end_compute_phase()

                _rd_Ab(tick)
                _ld_Bl(tick, kiter+3)

                begin_compute_phase()
                do_gemm(frag_C_bl, frag_B_l, frag_A_t, None, None)
                end_compute_phase()

                _ld_Br(tick, kiter+3)
                rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vm_load_cnt_a + vm_load_cnt_b*2))
                
                begin_compute_phase()
                do_gemm(frag_C_br, frag_B_r, frag_A_t, None, None)
                end_compute_phase()
            results = yield [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
                
            # else:

            #     # ===== 单缓冲 A + ping-pong B（象限序 TL,TR,BL,BR）；scale 按象限组 lazy 加载 =====
            #     # 每组仅持 4(scaleA)+2(scaleB) 个 SSA，避免像预取 16 个那样长期占用寄存器。
            #     # buf0 = tile kidx
            #     if const_expr(with_scale):
            #         sA_t0 = _load_sA(sA_top_baseM, kiter)
            #         sB_l0 = _load_sB(nb_l, kiter)
            #         sB_r0 = _load_sB(nb_r, kiter)
            #     else:
            #         sA_t0 = sB_l0 = sB_r0 = None
            #     rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
            #     rocdl.s_barrier()
            #     fx.copy(lds_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
            #     fx.copy(lds_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
            #     fx.copy(lds_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
            #     rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            #     begin_compute_phase()
            #     do_gemm(frag_C_tl, frag_B_l, frag_A_t, sA_t0, sB_l0)
            #     do_gemm(frag_C_tr, frag_B_r, frag_A_t, sA_t0, sB_r0)
            #     end_compute_phase()
            #     if const_expr(with_scale):
            #         sA_b0 = _load_sA(sA_bot_baseM, kiter)
            #     else:
            #         sA_b0 = None
            #     fx.copy(lds_copy_atom, s2r_src0_A_b, dest_frag_A_t, pred=None)
            #     rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
            #     begin_compute_phase()
            #     do_gemm(frag_C_bl, frag_B_l, frag_A_t, sA_b0, sB_l0)
            #     do_gemm(frag_C_br, frag_B_r, frag_A_t, sA_b0, sB_r0)
            #     end_compute_phase()
            #     rocdl.s_barrier()
            #     fx.copy(async_copy_atom, bL_g[None, None, None, kiter + 2], bL_s[0])
            #     fx.copy(async_copy_atom, aT_g[None, None, None, kiter + 2], aT_s[0])
            #     fx.copy(async_copy_atom, aB_g[None, None, None, kiter + 2], aB_s[0])
            #     fx.copy(async_copy_atom, bR_g[None, None, None, kiter + 2], bR_s[0])

            #     # buf1 = tile kidx+1
            #     if const_expr(with_scale):
            #         sA_t1 = _load_sA(sA_top_baseM, kiter + 1)
            #         sB_l1 = _load_sB(nb_l, kiter + 1)
            #         sB_r1 = _load_sB(nb_r, kiter + 1)
            #     else:
            #         sA_t1 = sB_l1 = sB_r1 = None
            #     rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
            #     rocdl.s_barrier()
            #     fx.copy(lds_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
            #     fx.copy(lds_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
            #     fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            #     rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            #     begin_compute_phase()
            #     do_gemm(frag_C_tl, frag_B_l, frag_A_t, sA_t1, sB_l1)
            #     do_gemm(frag_C_tr, frag_B_r, frag_A_t, sA_t1, sB_r1)
            #     end_compute_phase()
            #     if const_expr(with_scale):
            #         sA_b1 = _load_sA(sA_bot_baseM, kiter + 1)
            #     else:
            #         sA_b1 = None
            #     fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_t, pred=None)
            #     rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
            #     begin_compute_phase()
            #     do_gemm(frag_C_bl, frag_B_l, frag_A_t, sA_b1, sB_l1)
            #     do_gemm(frag_C_br, frag_B_r, frag_A_t, sA_b1, sB_r1)
            #     end_compute_phase()
            #     rocdl.s_barrier()
            #     fx.copy(async_copy_atom, bL_g[None, None, None, kiter + 3], bL_s[1])
            #     fx.copy(async_copy_atom, aT_g[None, None, None, kiter + 3], aT_s[1])
            #     fx.copy(async_copy_atom, aB_g[None, None, None, kiter + 3], aB_s[1])
            #     fx.copy(async_copy_atom, bR_g[None, None, None, kiter + 3], bR_s[1])
            #     results = yield [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]

        frag_C_tl.store(results[0])
        frag_C_tr.store(results[1])
        frag_C_bl.store(results[2])
        frag_C_br.store(results[3])

        # # ---- 尾部 2 个 k-tile：无 g2s，只 s2r + gemm；scale 按象限组 lazy 加载 ----
        # _kbt0 = fx.Int32(num_tiles - 2)
        # _kbt1 = fx.Int32(num_tiles - 1)
        # # buf0 = tile num_tiles-2（单缓冲 A，load-at-use）
        # if const_expr(with_scale):
        #     tA_t0 = _load_sA(sA_top_baseM, _kbt0)
        #     tB_l0 = _load_sB(nb_l, _kbt0)
        #     tB_r0 = _load_sB(nb_r, _kbt0)
        # else:
        #     tA_t0 = tB_l0 = tB_r0 = None
        # rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
        # rocdl.s_barrier()
        # fx.copy(lds_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
        # fx.copy(lds_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
        # fx.copy(lds_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
        # rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        # begin_compute_phase()
        # do_gemm(frag_C_tl, frag_B_l, frag_A_t, tA_t0, tB_l0)
        # do_gemm(frag_C_tr, frag_B_r, frag_A_t, tA_t0, tB_r0)
        # end_compute_phase()
        # if const_expr(with_scale):
        #     tA_b0 = _load_sA(sA_bot_baseM, _kbt0)
        # else:
        #     tA_b0 = None
        # fx.copy(lds_copy_atom, s2r_src0_A_b, dest_frag_A_t, pred=None)
        # rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
        # begin_compute_phase()
        # do_gemm(frag_C_bl, frag_B_l, frag_A_t, tA_b0, tB_l0)
        # do_gemm(frag_C_br, frag_B_r, frag_A_t, tA_b0, tB_r0)
        # end_compute_phase()

        # # buf1 = tile num_tiles-1
        # if const_expr(with_scale):
        #     tA_t1 = _load_sA(sA_top_baseM, _kbt1)
        #     tB_l1 = _load_sB(nb_l, _kbt1)
        #     tB_r1 = _load_sB(nb_r, _kbt1)
        # else:
        #     tA_t1 = tB_l1 = tB_r1 = None
        # rocdl.s_barrier()
        # fx.copy(lds_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
        # fx.copy(lds_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
        # fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
        # rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
        # begin_compute_phase()
        # do_gemm(frag_C_tl, frag_B_l, frag_A_t, tA_t1, tB_l1)
        # do_gemm(frag_C_tr, frag_B_r, frag_A_t, tA_t1, tB_r1)
        # end_compute_phase()
        # if const_expr(with_scale):
        #     tA_b1 = _load_sA(sA_bot_baseM, _kbt1)
        # else:
        #     tA_b1 = None
        # fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_t, pred=None)
        # rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0, lgkmcnt=0))
        # begin_compute_phase()
        # do_gemm(frag_C_bl, frag_B_l, frag_A_t, tA_b1, tB_l1)
        # do_gemm(frag_C_br, frag_B_r, frag_A_t, tA_b1, tB_r1)
        # end_compute_phase()

        if wave_id < 4:
            rocdl.s_barrier()

        # ---- epilogue store ----
        if const_expr(permlane_epilogue and TILE_N % 256 == 0):
            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
            lane_id = tid % 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4
            lane_group = lane_id // 16
            fragment_mode_0_repeat = TILE_N // 128
            fragment_mode_1_repeat = TILE_M // 64

            def store_c_quadrant(c_frag, quadrant_m, quadrant_n):
                for row_repeat in range_constexpr(fragment_mode_1_repeat):
                    for col_repeat in range_constexpr(0, fragment_mode_0_repeat, 2):
                        acc_a = Vec(c_frag[None, col_repeat, row_repeat].load())
                        acc_b = Vec(c_frag[None, col_repeat + 1, row_repeat].load())
                        d0_a = rocdl.cvt_pk_bf16_f32(acc_a[0], acc_a[1])
                        d1_a = rocdl.cvt_pk_bf16_f32(acc_a[2], acc_a[3])
                        d0_b = rocdl.cvt_pk_bf16_f32(acc_b[0], acc_b[1])
                        d1_b = rocdl.cvt_pk_bf16_f32(acc_b[2], acc_b[3])
                        swap0 = rocdl.permlane16_swap(pair_type, arith._to_raw(d0_a), arith._to_raw(d0_b), False, False)
                        swap1 = rocdl.permlane16_swap(pair_type, arith._to_raw(d1_a), arith._to_raw(d1_b), False, False)
                        packed = Vec.from_elements(
                            [
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                                fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                            ],
                            fx.Int32,
                        )
                        row = (
                            bid_x * TILE_M
                            + quadrant_m * (TILE_M // 2)
                            + row_repeat * 32
                            + wave_m * 16
                            + lane_id % 16
                        )
                        col = (
                            bid_y * TILE_N
                            + quadrant_n * (TILE_N // 2)
                            + col_repeat * 64
                            + lane_group % 2 * 64
                            + wave_n * 16
                            + lane_group // 2 * 8
                        )
                        byte_offset = (row * N + col) * 2
                        fx.buffer_ops.buffer_store(packed, c_store_rsrc, byte_offset, offset_is_bytes=True)

            store_c_quadrant(frag_C_tl, 0, 0)
            store_c_quadrant(frag_C_tr, 0, 1)
            store_c_quadrant(frag_C_bl, 1, 0)
            store_c_quadrant(frag_C_br, 1, 1)
        else:
            c_frag_bf16 = fx.make_fragment_like(frag_C_tl, dtype=fx.BFloat16)
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            store_thr = fx.make_tiled_copy_C(store_atom, tiled_mma).get_slice(tid)

            def store_c_quadrant(c_frag, bC):
                c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
                fx.copy(store_atom, store_thr.retile(c_frag_bf16), store_thr.partition_D(bC))

            store_c_quadrant(frag_C_tl, bC_tl)
            store_c_quadrant(frag_C_tr, bC_tr)
            store_c_quadrant(frag_C_bl, bC_bl)
            store_c_quadrant(frag_C_br, bC_br)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, scaleA: fx.Tensor, scaleB: fx.Tensor,
                    M: int, stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C, scaleA, scaleB, M).launch(
            grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1), block=(512, 1, 1), stream=stream
        )

    return launch_gemm


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
M = int(os.environ.get("GEMM_M", 8192))
N = int(os.environ.get("GEMM_N", 8192))
K = int(os.environ.get("GEMM_K", 8192))
PERMLANE_EPILOGUE = _env_flag("PERMLANE", "1")

import pyhip


def _load_shuffle_weight():
    import sys as _sys, os.path as _osp
    _root = _osp.abspath(_osp.join(_osp.dirname(__file__), "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from tests.utils import shuffle_weight
    return shuffle_weight


def run_test(M, N, K, perf=False, permlane_output=True, preshuffle_b=False, with_scale=False,
             run_count=50, data_clones=32):
    shuffle_weight = _load_shuffle_weight() if preshuffle_b else None

    def _shuffle_b(x):
        return shuffle_weight(x, layout=(16, 64)) if preshuffle_b else x

    KB = K // 128
    empty = torch.empty(0, device="cuda", dtype=torch.float32)

    def _gen_scales():
        if not with_scale:
            return empty, empty
        sA = torch.rand((M, KB), device="cuda", dtype=torch.float32)
        sB = torch.rand((N // 128, KB), device="cuda", dtype=torch.float32)
        return sA, sB

    def _ref(a, b, sA, sB):
        if not with_scale:
            return a.float() @ b.float().t()
        a_deq = (a.float().view(M, KB, 128) * sA.view(M, KB, 1)).view(M, K)
        b_deq = (b.float().view(N // 128, 128, KB, 128) * sB.view(N // 128, 1, KB, 1)).reshape(N, K)
        return a_deq @ b_deq.t()

    a = (torch.rand(M, K, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    b = (torch.rand(N, K, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    sA, sB = _gen_scales()
    ref = _ref(a, b, sA, sB)
    out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    weight = _shuffle_b(b)
    stream = torch.cuda.current_stream()
    args = (a.view(torch.int8).view(-1), weight.view(torch.int8).view(-1), out.view(-1),
            sA.view(-1), sB.view(-1), M, stream)

    launcher = compile_gemm_fp8_8wave(TILE_M, TILE_N, TILE_K, N, K, permlane_epilogue=permlane_output,
                                      preshuffle_b=preshuffle_b, with_scale=with_scale)
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    abs_err = (out.float() - ref).abs()
    rel = abs_err / (ref.abs() + 1e-3)
    atol = 0.02 * ref.abs().max().item() + 0.01
    is_correct = torch.allclose(out.float(), ref, rtol=0.05, atol=atol)
    print(f"####M={M} N={N} K={K} 8wave preshuffle_b={preshuffle_b} with_scale={with_scale} "
          f"is_correct={is_correct} max_abs={abs_err.max().item():.3f} max_rel={rel.max().item():.3f}")

    if not perf:
        return is_correct

    As = [torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn) for _ in range(data_clones)]
    Bs = [_shuffle_b(torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)) for _ in range(data_clones)]
    SAs = [(_gen_scales()[0] if with_scale else empty) for _ in range(data_clones)]
    SBs = [(_gen_scales()[1] if with_scale else empty) for _ in range(data_clones)]
    Cs = [torch.zeros((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(data_clones)]
    arg_sets = [
        (As[i].view(torch.int8).view(-1), Bs[i].view(torch.int8).view(-1), Cs[i].view(-1),
         SAs[i].view(-1), SBs[i].view(-1), M, stream)
        for i in range(data_clones)
    ]
    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 1 + M * N * 2
    for i in range(data_clones):
        kernel(*arg_sets[i])
    torch.cuda.synchronize()
    di = 0
    latencies = []
    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"gemm_{di}") as p:
            kernel(*arg_sets[di])
        latencies.append(p.dt_ms)
    latencies.sort()
    best_ms = latencies[0]
    print(f"\n=== perf 8wave M={M} N={N} K={K} with_scale={with_scale} ===")
    print(f"gemm:  {best_ms*1e3:.1f} us  {flops/(best_ms*1e-3)/1e12:.2f} TFLOPS  {mem_bytes/(best_ms*1e-3)/1e9:.1f} GB/s")
    return is_correct


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "fp8 MFMA_Scale 需要 gfx950"
    torch.manual_seed(0)
    run_test(M=8192, N=8192, K=8192, perf=False, permlane_output=PERMLANE_EPILOGUE, with_scale=False)
    run_test(M=8192, N=8192, K=8192, perf=True, permlane_output=PERMLANE_EPILOGUE, with_scale=False)
