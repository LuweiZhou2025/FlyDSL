# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# fp8 GEMM (C = B * A, 输出 bf16)，按 test_gemm_v9.py 的方式用 tile + layout 抽象编写
# （flat_divide / make_tiled_copy / make_tiled_mma / make_fragment / fx.copy / fx.gemm），
# 不做手动 byte-offset 计算。
#   - BLOCK_M=BLOCK_N=BLOCK_K=128, TILE_M=TILE_N=256, 4-wave, 2x2 quadrant
#   - MFMA 指令 V_MFMA_SCALE_F32_16X16X128_F8F6F4（scale=0 => 不含 scale）
#   - A/B 均普通输入 + LDS bank-conflict 消解：padding（默认，[[1024,32]] 单 padding，对标 bf16 v9）
#     或 swizzle（lds_swizzle=True, MBase=4）。LDS ping-pong 双缓冲 + 寄存器软件流水。
#   - 约定：A 走 make_fragment_B，B 走 make_fragment_A；fx.gemm(mma, C, frag_B, frag_A)。
#   - 用 SWIZZLE=1 环境变量切到 swizzle 版本。
#
# 运行：cd /mywork/FlyDSL/tests/kernels && HIP_VISIBLE_DEVICES=4 python ./test_gemm_v9_fp8.py

import os
import math

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


def waitvmcnt_barrier(vmcnt):
    # 对标 test_gemm_v9.py：s_waitcnt vmcnt(n) + s_waitcnt lgkmcnt(0) + s_barrier，
    # 一次完成 vmem/lds 等待与全 block 同步（内含 s_barrier，无需再单独 gpu.barrier）。
    rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
    rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
    rocdl.s_barrier()


def hot_loop_scheduler_mainloop(group_id, vmem_ops, dsrd_ops):
    
    total_mfmas = 16
    remaing_mfmas = total_mfmas - vmem_ops*2 - dsrd_ops
    for _ in range_constexpr(dsrd_ops):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, group_id)
    for _ in range_constexpr(vmem_ops):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    for _ in range_constexpr(remaing_mfmas):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)



def compile_gemm_fp8(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    pid_swizzle=True,
    lds_swizzle=False,
):
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
    element_type = fx.Float8E4M3FN
    elements_per_128b = 16  # 128bit / fp8(8bit)
    # 主循环 vmcnt 阈值：fp8 数据量与 bf16 一致（fp8 1B×BLOCK_K128 == bf16 2B×BLOCK_K64），
    # 每块/每 array 的 128b buffer_load 条数与 bf16 相同，故 vmcnt 直接对标 bf16 gemm_v9 = 20。
    _VMCNT = int(os.environ.get("VMCNT", "20"))
    # sched_group_barrier 精确调度（对标 bf16 hot_loop_scheduler_mainloop）：
    # fp8 每象限一条 MFMA(16x16x128) 抵 bf16 两条(16x16x32)，故 MFMA 计数减半（32->16）；
    # ds_read/vmem 计数不变（数据量一致）。仅在完整流水迭代（同时有 s2r+g2s）时应用。
    _USE_SCHED = _env_flag("SCHED", "1")

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

    # A padding（对标 bf16 v9 / gluon a8w8 kWidth=16 单 padding [[1024,32]]：每 8 行 pad 32 fp8=32B）
    A_PAD = 32
    A_GROUP = 8 * BLOCK_K + A_PAD  # 1056
    a_lds_elems = (BLOCK_M // 8) * A_GROUP  # 16*1056 = 16896
    b_lds_elems = (BLOCK_N // 8) * A_GROUP  # 16896

    @fx.struct
    class LDS:
        a_t0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gemm_kernel(argA: fx.Tensor, argB: fx.Tensor, argC: fx.Tensor, M: int):
        tid = fx.thread_idx.x
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

        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 0, None]
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x * 2 + 1, None]
        bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 0, None]
        bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y * 2 + 1, None]
        # A/B 全局 tile 视图：swizzle 版（全局 swizzle）或 padding 版（分组）。
        if const_expr(lds_swizzle):
            _nb = 4  # fp8 128-bit = 16 elem = 2^4
            _swg = fx.static(fx.SwizzleType.get(3, _nb, K.bit_length() - 1 - _nb))
            bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), fx.make_composed_layout(_swg, fx.get_layout(bA_t))))
            bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), fx.make_composed_layout(_swg, fx.get_layout(bA_b))))
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), fx.make_composed_layout(_swg, fx.get_layout(bB_l))))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), fx.make_composed_layout(_swg, fx.get_layout(bB_r))))
        else:
            # A: 分组全局视图（每 8 行为一组，映射到 padding LDS 的行组），对标 bf16 v9 bA_layout。
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

        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 0]
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 0, bid_y * 2 + 1]
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 0]
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x * 2 + 1, bid_y * 2 + 1]

        # ---- tiled MMA: MFMA_Scale 16x16x128 f8f6f4, scale=0 (no scale) ----
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, element_type))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
        k_perm = fx.make_layout((32, 4), (1, 32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)), (None, None, k_perm))
        thr_mma = tiled_mma.thr_slice(tid)

        # ---- copy atoms ----
        async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)

        # ---- LDS 分配 ----
        lds = fx.SharedAllocator().allocate(LDS).peek()

        # wr/rd LDS 布局 + DMA tiled copy：swizzle 版（ordered wr + swizzle rd + make_layout_tv DMA）
        # 或 padding 版（分组 wr/rd + 专用 a_dma）。A/B 共用同一 dma。
        if const_expr(lds_swizzle):
            _swl = fx.static(fx.SwizzleType.get(3, 4, BLOCK_K.bit_length() - 1 - 4))
            _wr = fx.make_ordered_layout((BLOCK_M, BLOCK_K), (1, 0))
            _rd = fx.make_composed_layout(_swl, _wr)
            _g2s_tile, _g2s_tv = fx.make_layout_tv(
                fx.make_layout((8 * 4, 8), (8, 1)),
                fx.make_layout((1, elements_per_128b), (1, 1)),
            )
            dma = fx.make_tiled_copy(buffer_copy_atom, _g2s_tv, _g2s_tile).get_slice(tid)
        else:
            _wr = fx.make_layout(((8, 16), BLOCK_K), ((BLOCK_K, A_GROUP), 1))
            _rd = fx.make_layout(((16, 8), (32, BLOCK_K // 32)), ((A_GROUP, BLOCK_K), (1, 32)))
            _a_dma_tv = fx.make_layout(
                ((8, 8, 4), elements_per_128b),
                ((elements_per_128b * 32, 1, 8), 32),
            )
            dma = fx.make_tiled_copy(buffer_copy_atom, _a_dma_tv, fx.make_tile(32, BLOCK_K)).get_slice(tid)

        sA_t_wr = [fx.make_view(lds.a_t0.ptr, _wr), fx.make_view(lds.a_t1.ptr, _wr)]
        sA_b_wr = [fx.make_view(lds.a_b0.ptr, _wr), fx.make_view(lds.a_b1.ptr, _wr)]
        sA_t_rd = [fx.make_view(lds.a_t0.ptr, _rd), fx.make_view(lds.a_t1.ptr, _rd)]
        sA_b_rd = [fx.make_view(lds.a_b0.ptr, _rd), fx.make_view(lds.a_b1.ptr, _rd)]
        sB_l_wr = [fx.make_view(lds.b_l0.ptr, _wr), fx.make_view(lds.b_l1.ptr, _wr)]
        sB_r_wr = [fx.make_view(lds.b_r0.ptr, _wr), fx.make_view(lds.b_r1.ptr, _wr)]
        sB_l_rd = [fx.make_view(lds.b_l0.ptr, _rd), fx.make_view(lds.b_l1.ptr, _rd)]
        sB_r_rd = [fx.make_view(lds.b_r0.ptr, _rd), fx.make_view(lds.b_r1.ptr, _rd)]

        aT_g = dma.partition_S(bA_t)
        aB_g = dma.partition_S(bA_b)
        bL_g = dma.partition_S(bB_l)
        bR_g = dma.partition_S(bB_r)
        aT_s = [dma.partition_D(sA_t_wr[0]), dma.partition_D(sA_t_wr[1])]
        aB_s = [dma.partition_D(sA_b_wr[0]), dma.partition_D(sA_b_wr[1])]
        bL_s = [dma.partition_D(sB_l_wr[0]), dma.partition_D(sB_l_wr[1])]
        bR_s = [dma.partition_D(sB_r_wr[0]), dma.partition_D(sB_r_wr[1])]

        # ---- LDS -> reg（对标 gemm_v9：A 走 B-operand，B 走 A-operand；均 padding rd）----
        # 每个 slice 只有一份寄存器 fragment（无寄存器双缓冲），双缓冲仅在 LDS 层（buf0/buf1）。
        copy_a = fx.make_tiled_copy_B(lds_copy_atom, tiled_mma).get_slice(tid)
        copy_b = fx.make_tiled_copy_A(lds_copy_atom, tiled_mma).get_slice(tid)
        # s2r 源：LDS buf0 / buf1（对标 gemm_v9 的 s2r_src0_* / s2r_src1_*）
        s2r_src0_A_t = copy_a.partition_S(sA_t_rd[0])
        s2r_src0_A_b = copy_a.partition_S(sA_b_rd[0])
        s2r_src0_B_l = copy_b.partition_S(sB_l_rd[0])
        s2r_src0_B_r = copy_b.partition_S(sB_r_rd[0])
        s2r_src1_A_t = copy_a.partition_S(sA_t_rd[1])
        s2r_src1_A_b = copy_a.partition_S(sA_b_rd[1])
        s2r_src1_B_l = copy_b.partition_S(sB_l_rd[1])
        s2r_src1_B_r = copy_b.partition_S(sB_r_rd[1])

        # 单份寄存器 fragment（A -> make_fragment_B, B -> make_fragment_A）
        frag_A_t = thr_mma.make_fragment_B(sA_t_rd[0])
        frag_A_b = thr_mma.make_fragment_B(sA_b_rd[0])
        frag_B_l = thr_mma.make_fragment_A(sB_l_rd[0])
        frag_B_r = thr_mma.make_fragment_A(sB_r_rd[0])
        dest_frag_A_t = copy_a.retile(frag_A_t)
        dest_frag_A_b = copy_a.retile(frag_A_b)
        dest_frag_B_l = copy_b.retile(frag_B_l)
        dest_frag_B_r = copy_b.retile(frag_B_r)

        # ---- C fragments（对标 test_gemm fp8: make_fragment_C 后 select[0,2,1]）----
        frag_C_tl = fx.select(thr_mma.make_fragment_C(bC_tl), [0, 2, 1])
        frag_C_tr = fx.select(thr_mma.make_fragment_C(bC_tr), [0, 2, 1])
        frag_C_bl = fx.select(thr_mma.make_fragment_C(bC_bl), [0, 2, 1])
        frag_C_br = fx.select(thr_mma.make_fragment_C(bC_br), [0, 2, 1])

        num_tiles = K // BLOCK_K
        assert num_tiles >= 4

        # ---- prologue：预取 tile0/tile1 到 LDS buf0/buf1，再把 buf0 的 B_l/A_t s2r 到寄存器 ----
        # 对标 gemm_v9：8 条 async g2s（2 tile × 4 array），waitvmcnt_barrier(24)，再 s2r B_l/A_t。
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

        do_g2s(0, 0)
        do_g2s(1, 1)

        waitvmcnt_barrier(24)
        fx.copy(lds_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
        fx.copy(lds_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
        rocdl.sched_barrier(0)

        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)
        rocdl.sched_barrier(0)

        # ---- 主循环（对标 gemm_v9：单份 fragment，一次迭代吃 2 个 k-tile = 8 个 region）----
        # 每个 region：1 个象限 fx.gemm(C=B*A) + 下一 operand 的 s2r + 再下一块的 g2s，
        # 用 s2r_src0_*/s2r_src1_* 在 LDS buf0/buf1 之间 ping-pong；每个 slice 顺序与 gemm_v9 一致。
        # k-tile 内 4 个象限的顺序固定为：tl(A_t·B_l) -> bl(A_b·B_l) -> tr(A_t·B_r) -> br(A_b·B_r)。
        for kidx in range_constexpr(0, num_tiles - 2, 2):
            kiter = fx.Int32(kidx)

            # ---- k-tile = buf0：4 象限 ----
            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_tl, frag_B_l, frag_A_t, frag_C_tl)
            fx.copy(lds_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
            fx.copy(async_copy_atom, bL_g[None, None, None, kiter + 2], bL_s[0])
            hot_loop_scheduler_mainloop(0, 4, 8)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_bl, frag_B_l, frag_A_b, frag_C_bl)
            fx.copy(lds_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
            fx.copy(async_copy_atom, aT_g[None, None, None, kiter + 2], aT_s[0])
            hot_loop_scheduler_mainloop(1, 4, 8)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_tr, frag_B_r, frag_A_t, frag_C_tr)
            fx.copy(lds_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
            fx.copy(async_copy_atom, aB_g[None, None, None, kiter + 2], aB_s[0])
            hot_loop_scheduler_mainloop(2, 4, 8)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_br, frag_B_r, frag_A_b, frag_C_br)
            fx.copy(lds_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
            fx.copy(async_copy_atom, bR_g[None, None, None, kiter + 2], bR_s[0])
            hot_loop_scheduler_mainloop(3, 4, 8)
            rocdl.sched_barrier(0)

            # ---- k-tile = buf1：4 象限 ----
            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_tl, frag_B_l, frag_A_t, frag_C_tl)
            fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
            fx.copy(async_copy_atom, bL_g[None, None, None, kiter + 3], bL_s[1])
            hot_loop_scheduler_mainloop(4, 4, 8)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_bl, frag_B_l, frag_A_b, frag_C_bl)
            fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            fx.copy(async_copy_atom, aT_g[None, None, None, kiter + 3], aT_s[1])
            hot_loop_scheduler_mainloop(5, 4, 8)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_tr, frag_B_r, frag_A_t, frag_C_tr)
            fx.copy(lds_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
            fx.copy(async_copy_atom, aB_g[None, None, None, kiter + 3], aB_s[1])
            hot_loop_scheduler_mainloop(6, 4, 8)
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            fx.gemm(mma_atom, frag_C_br, frag_B_r, frag_A_b, frag_C_br)
            fx.copy(lds_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
            fx.copy(async_copy_atom, bR_g[None, None, None, kiter + 3], bR_s[1])
            hot_loop_scheduler_mainloop(7, 4, 8)
            rocdl.sched_barrier(0)

        # ---- epilogue：最后 2 个 k-tile（buf0 / buf1），无 g2s，只做 s2r + gemm ----
        # buf0 的 4 象限
        waitvmcnt_barrier(20)
        fx.gemm(mma_atom, frag_C_tl, frag_B_l, frag_A_t, frag_C_tl)
        fx.copy(lds_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
        hot_loop_scheduler_mainloop(0, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(16)
        fx.gemm(mma_atom, frag_C_bl, frag_B_l, frag_A_b, frag_C_bl)
        fx.copy(lds_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
        hot_loop_scheduler_mainloop(1, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(12)
        fx.gemm(mma_atom, frag_C_tr, frag_B_r, frag_A_t, frag_C_tr)
        fx.copy(lds_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
        hot_loop_scheduler_mainloop(2, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(8)
        fx.gemm(mma_atom, frag_C_br, frag_B_r, frag_A_b, frag_C_br)
        fx.copy(lds_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
        hot_loop_scheduler_mainloop(3, 0, 8)
        rocdl.sched_barrier(0)

        # buf1 的 4 象限（无更多 s2r，直接算完）
        waitvmcnt_barrier(4)
        fx.gemm(mma_atom, frag_C_tl, frag_B_l, frag_A_t, frag_C_tl)
        fx.copy(lds_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
        hot_loop_scheduler_mainloop(4, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(0)
        fx.gemm(mma_atom, frag_C_bl, frag_B_l, frag_A_b, frag_C_bl)
        fx.copy(lds_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
        hot_loop_scheduler_mainloop(5, 0, 8)
        rocdl.sched_barrier(0)

        fx.gemm(mma_atom, frag_C_tr, frag_B_r, frag_A_t, frag_C_tr)
        hot_loop_scheduler_mainloop(6, 0, 0)
        rocdl.sched_barrier(0)
        fx.gemm(mma_atom, frag_C_br, frag_B_r, frag_A_b, frag_C_br)
        hot_loop_scheduler_mainloop(7, 0, 0)
        rocdl.sched_barrier(0)

        # ---- store: f32 -> bf16（暂不与 gemm 交织）----
        # 注意：fp8 op1=B 走 make_fragment_A 槽，C 的 wave 朝向相对 bf16 转置，
        # 故 c_tv 的两个 wave 维 stride 需交换为 (512, 16)。
        store_atom_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        c_layout_w = fx.make_tiled_copy(
            store_atom_bf16,
            fx.make_layout(((16, 4, 2, 2), 4), ((1, 128, 512, 16), 32)),
            fx.make_tile(32, 32),
        )
        store_thr = c_layout_w.get_slice(tid)

        def store_quadrant(c_frag, bC):
            c_sel = fx.select(c_frag, [0, 2, 1])
            c_bf16 = fx.make_fragment_like(c_sel, dtype=fx.BFloat16)
            c_bf16.store(c_sel.load().to(fx.BFloat16))
            fx.copy(store_atom_bf16, store_thr.retile(c_bf16), store_thr.partition_D(bC))

        store_quadrant(frag_C_tl, bC_tl)
        store_quadrant(frag_C_tr, bC_tr)
        store_quadrant(frag_C_bl, bC_bl)
        store_quadrant(frag_C_br, bC_br)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, M: int, stream: fx.Stream = fx.Stream(None)):
        # 累加器钉到 AGPR（force-agpr）+ mfma-vgpr-form=False：避免 C 累加器 VGPR/AGPR 混放导致的
        # v_accvgpr 拷贝与 VGPR 压力（对标 test_gemm_v9.py）。
        value_attrs = {
            "rocdl.waves_per_eu": 1,
            "passthrough": [["amdgpu-agpr-alloc", "256,256"]],
        }
        gemm_kernel(A, B, C, M, value_attrs=value_attrs).launch(
            grid=(div_up(M, TILE_M) * div_up(N, TILE_N), 1, 1), block=(256, 1, 1), stream=stream
        )

    launch_gemm.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
    return launch_gemm


# =========================== test / perf ===========================
TILE_M = 256
TILE_N = 256
TILE_K = 128
M = int(os.environ.get("GEMM_M", 4096))
N = int(os.environ.get("GEMM_N", 4096))
K = int(os.environ.get("GEMM_K", 4096))


def _make_problem():
    a = torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)
    ref = a.float() @ b.float().t()
    out = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), out.view(-1), M, torch.cuda.current_stream())
    return out, ref, args


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "fp8 MFMA_Scale 需要 gfx950"
    torch.manual_seed(0)

    out, ref, args = _make_problem()
    launcher = compile_gemm_fp8(TILE_M, TILE_N, TILE_K, N, K, lds_swizzle=_env_flag("SWIZZLE"))
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    acc = torch.allclose(out.float(), ref, rtol=0.05, atol=0.5)
    print(f"fp8 v9-style  M={M} N={N} K={K}  BLOCK={TILE_M//2}x{TILE_N//2}x{TILE_K}  is_correct={acc}")
    if not acc:
        mism = (out.float() - ref).abs()
        print(f"  max_abs_err={mism.max().item():.3f}  mism_count={(mism>0.5).sum().item()}/{M*N}")

    # ---- perf（对标 test_gemm_v9.py compare_perf：pyhip.cudaPerf + 多份数据轮转）----
    # 多份数据轮转：A/B/C 各 data_clones 份，轮流喂入。单份 A+B 只有几十 MB，反复喂同一份
    # 会常驻 L2 -> 高估 TFLOPS；轮转多份（远大于 L2）确保每次都是 cold data，排除 cache 影响。
    import pyhip

    data_clones = 32
    run_count = 50
    As = [torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn) for _ in range(data_clones)]
    Bs = [torch.randint(-2, 3, (N, K), device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn) for _ in range(data_clones)]
    Cs = [torch.zeros((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(data_clones)]
    stream = torch.cuda.current_stream()
    arg_sets = [
        (As[i].view(torch.int8).view(-1), Bs[i].view(torch.int8).view(-1), Cs[i].view(-1), M, stream)
        for i in range(data_clones)
    ]

    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 1 + M * N * 2  # fp8 A+B (1B) + bf16 C (2B)

    # warmup（轮转，把所有 clone 都碰一遍）
    for i in range(data_clones):
        kernel(*arg_sets[i])
    torch.cuda.synchronize()

    # 每次测一个 kernel launch，轮转 clone，取最优（best）延迟
    di = 0
    latencies = []
    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"gemm_{di}") as p:
            kernel(*arg_sets[di])
        latencies.append(p.dt_ms)
    latencies.sort()
    best_ms = latencies[0]
    tflops = flops / (best_ms * 1e-3) / 1e12
    bw_gbs = mem_bytes / (best_ms * 1e-3) / 1e9
    print(f"  perf(best, {data_clones} clones): {best_ms*1e3:8.1f} us   {tflops:8.1f} TFLOPS   {bw_gbs:8.1f} GB/s")
