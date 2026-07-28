# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch
import math
import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.compiler as flyc
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector, arith
import os
from flydsl._mlir.dialects import llvm as _llvm

from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr.typing import T as _T
from flydsl.compiler.ast_rewriter import ASTRewriter

def hot_loop_scheduler_mainloop():
    rocdl.sched_mfma(4)
    for _ in range_constexpr(8):
        rocdl.sched_dsrd(1)
        rocdl.sched_mfma(1)
    for _ in range_constexpr(4):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(4)
    rocdl.sched_mfma(4)



# every 8 contineous row pad 16 elements. (need 128/8-1) * 16 elements padding totally.
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")



def enable_dump_ir(enable_debug_info=True):
    if enable_debug_info:
        import flydsl
        from flydsl.utils.env import DebugEnvManager
        from flydsl._mlir import ir

        DebugEnvManager.enable_debug_info = enable_debug_info
        DebugEnvManager.dump_asm = True
        DebugEnvManager.dump_ir = True
        DebugEnvManager.dump_dir = "my_ir_dumps"
        ir._globals.register_traceback_file_inclusion(__file__)
        ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
        ir._globals.set_loc_tracebacks_frame_limit(40)
        ir._globals.set_loc_tracebacks_enabled(True)
        os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def encode_waitcnt_950(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier\n",
        # asm_string=f"s_waitcnt vmcnt({count})\ns_barrier\ns_waitcnt lgkmcnt(0)\n",
        constraints="",
        has_side_effects=True,
    )

def waitvmcnt_barrier(vmcnt):
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        rocdl.s_barrier()

class Mfma16x16x64:
    def __init__(self, n_tiles_a, n_tiles_b, mma_atom, use_inline_asm=True):
        self.mma_atom = mma_atom
        # self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA(16, 16, 32, fx.BFloat16))
        # self.accum_type = Vec.make_type(4, fx.Float32)
        # self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.use_inline_asm = use_inline_asm
    
    def _do_mma(self, a, b, c):
        # a, b, c are register-memref fragment slices; load them to vectors first.
        a_i32x4 = vector.bitcast(_T.vec(4, _T.i32), a.load())
        b_i32x4 = vector.bitcast(_T.vec(4, _T.i32), b.load())
        c_vec = c.load()
        res_ty = _T.vec(4, _T.f32)
        return _llvm.inline_asm(
            res_ty,
            [arith._to_raw(a_i32x4), arith._to_raw(b_i32x4), arith._to_raw(c_vec)],
            "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0",
            "=a,v,v,0",
            has_side_effects=False,
        )

    def call_BxA(self, aa, bb, c):
        # assert len(a) == self.n_tiles_a
        # assert len(b) == self.n_tiles_b
        # assert len(c) == self.n_tiles_a * self.n_tiles_b
        if self.use_inline_asm:
            """Use native K32 MFMA ops with explicit per-slice accumulator chains."""
            for n in range_constexpr(self.n_tiles_b):
                for m in range_constexpr(self.n_tiles_a):
                    c_slice = c[None, n, m]
                    acc = arith._to_raw(c_slice.load())
                    for k in range_constexpr(2):
                        a = bb[None, n, k].load()
                        b = aa[None, m, k].load()
                        acc = rocdl.mfma_f32_16x16x32_bf16(
                            T.vec(4, T.f32),
                            [arith._to_raw(a), arith._to_raw(b), arith._to_raw(acc), 0, 0, 0],
                        )
                    c_slice.store(acc)
            # for i in range_constexpr(self.n_tiles_a):
            #     for j in range_constexpr(self.n_tiles_b):
            #         c[None, j, i] = self._do_mma(b[None, j, 0], a[None, i, 0], c[None, j, i])
            #         c[None, j, i] = self._do_mma(b[None, j, 1], a[None, i, 1], c[None, j, i])
        else:
            fx.gemm(self.mma_atom, c, b, a, c)



def div_up(x, y):
    return (x + y - 1) // y

def compile_gemm(
    TILE_M,
    TILE_N,
    TILE_K,
    N,
    K,
    dtype="bf16",
    pin_bf16_agpr=True,
    lds_swizzle=False,
    pid_swizzle=False,
    permlane_epilogue=True,
):
    BLOCK_M = TILE_M // 2
    BLOCK_N = TILE_N // 2
    BLOCK_K = TILE_K
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

    PADDING_ELEMS = 16
    PADDING_NUM = PADDING_ELEMS * (16 - 1)
    if lds_swizzle:
        PADDING_NUM = 0
    @fx.struct
    class LDS_PADDING:
        lds0_a_t: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds0_a_b: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds0_b_l: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
        lds0_b_r: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
        lds1_a_t: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds1_a_b: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
        lds1_b_l: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
        lds1_b_r: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]
    

    element_type = fx.BFloat16

    @flyc.kernel
    def gemm_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argC: fx.Tensor,
        M: int
    ):
        tid = fx.thread_idx.x
        num_pid_n = div_up(N, TILE_N)
        if const_expr(pid_swizzle):
            bid_x, bid_y = get_pids_950(fx.block_idx.x, M, fx.grid_dim.x, 8, 4)
        else:
            bid_x = fx.block_idx.x // num_pid_n
            bid_y = fx.block_idx.x % num_pid_n


        a_iter = fx.get_iter(argA)
        b_iter = fx.get_iter(argB)
        A_2d = fx.Tensor(fx.make_view(
            a_iter,
            fx.make_layout((M, K), (K, 1)),
        ))
        
        # A_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
        # B_2d = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))))
        # C_2d = fx.Tensor(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))))
        B_2d = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))))
        C_2d = fx.Tensor(fx.make_view(
            fx.get_iter(argC),
            fx.make_layout((M, N), (N, 1)),
        ))


        A = fx.rocdl.make_buffer_tensor(A_2d,  max_size=False)
        B = fx.rocdl.make_buffer_tensor(B_2d,  max_size=False)
        C = fx.rocdl.make_buffer_tensor(C_2d,  max_size=False)

        if lds_swizzle:
            num_base = 3
            num_bits = 3
            num_shift = K.bit_length() - 1 - num_base 
            
            GA_SWIZZLE_LAYOUT = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, num_shift)),
                fx.get_layout(A),
            )
            GB_SWIZZLE_LAYOUT = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, num_shift)),
                fx.get_layout(B),
            )
            # A =fx.make_view(fx.get_iter(A), GA_SWIZZLE_LAYOUT)
            # B =fx.make_view(fx.get_iter(B), GB_SWIZZLE_LAYOUT)
            

        bA_t = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x*2 + 0, None]  # (BM, BK, k)
        bA_b = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x*2 + 1, None]  # (BM, BK, k)
        bB_l = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y*2 + 0, None]  # (BN, BK, k)
        bB_r = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y*2 + 1, None]  # (BN, BK, k)
        
        bC_tl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 0, bid_y*2 + 0]  # (BM, BN)
        bC_tr = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 0, bid_y*2 + 1]  # (BM, BN)
        bC_bl = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 1, bid_y*2 + 0]  # (BM, BN)
        bC_br = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x*2 + 1, bid_y*2 + 1]  # (BM, BN)
        if not lds_swizzle:
            # A, B read layout
            bA_layout = fx.make_layout(((8, BLOCK_M//8), BLOCK_K, K//BLOCK_K), ((BLOCK_M//8*K, K), 1, BLOCK_K))
            bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), bA_layout))
            bA_b = fx.Tensor(fx.make_view(fx.get_iter(bA_b), bA_layout))
            bB_layout = fx.make_layout(((8, BLOCK_N//8), BLOCK_K, K//BLOCK_K), ((BLOCK_N//8*K, K), 1, BLOCK_K))
            bB_l = fx.Tensor(fx.make_view(fx.get_iter(bB_l), bB_layout))
            bB_r = fx.Tensor(fx.make_view(fx.get_iter(bB_r), bB_layout))

        # read and write LDS tensor view.
        lds_layout_rd =fx.make_layout(((16, 8), (32, 2)), ((512+PADDING_ELEMS, 64), (1, 32)))
        lds_layout_wr =fx.make_layout(((8, 16), 64), ((64, 8*64+PADDING_ELEMS), 1))
        if lds_swizzle:
            lds_layout_wr =fx.make_ordered_layout((BLOCK_M, BLOCK_K), (1, 0))
            lds_layout_rd = lds_layout_wr
            # lds_layout_rd = fx.make_composed_layout(
            #     fx.static(fx.SwizzleType.get(3, 3, 3)),
            #     lds_layout_wr,
            # )
        lds = fx.SharedAllocator().allocate(LDS_PADDING).peek()

        #LDS 0
        lds0_A_t_rd = fx.make_view(lds.lds0_a_t.ptr, lds_layout_rd)
        lds0_A_b_rd = fx.make_view(lds.lds0_a_b.ptr, lds_layout_rd)
        lds0_A_t_wr = fx.make_view(lds.lds0_a_t.ptr, lds_layout_wr)
        lds0_A_b_wr = fx.make_view(lds.lds0_a_b.ptr, lds_layout_wr)
        lds0_B_l_rd = fx.make_view(lds.lds0_b_l.ptr, lds_layout_rd)
        lds0_B_r_rd = fx.make_view(lds.lds0_b_r.ptr, lds_layout_rd)
        lds0_B_l_wr = fx.make_view(lds.lds0_b_l.ptr, lds_layout_wr)
        lds0_B_r_wr = fx.make_view(lds.lds0_b_r.ptr, lds_layout_wr)


        #LDS 1
        lds1_A_t_rd = fx.make_view(lds.lds1_a_t.ptr, lds_layout_rd)
        lds1_A_b_rd = fx.make_view(lds.lds1_a_b.ptr, lds_layout_rd)
        lds1_A_t_wr = fx.make_view(lds.lds1_a_t.ptr, lds_layout_wr)
        lds1_A_b_wr = fx.make_view(lds.lds1_a_b.ptr, lds_layout_wr)
        lds1_B_l_rd = fx.make_view(lds.lds1_b_l.ptr, lds_layout_rd)
        lds1_B_r_rd = fx.make_view(lds.lds1_b_r.ptr, lds_layout_rd)
        lds1_B_l_wr = fx.make_view(lds.lds1_b_l.ptr, lds_layout_wr)
        lds1_B_r_wr = fx.make_view(lds.lds1_b_r.ptr, lds_layout_wr)
        
        # copy atoms
        async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        lsd_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        buffer_copy_atom_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        buffer_copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        
        # DMA copy tiles
        ac_tile_mn = fx.make_tile(32, 64)
        ac_tv_layout =  fx.make_layout(((8, 8, 4), 8), ((8*4*8, 1, 8), 4*8))
        ac_tiled_copy = fx.make_tiled_copy(buffer_copy_atom_bf16, ac_tv_layout, ac_tile_mn)
        ac_thr = ac_tiled_copy.get_slice(tid)
        # DMA copy partition src, dest
        ac_src_A_t = ac_thr.partition_S(bA_t)
        ac_src_A_b = ac_thr.partition_S(bA_b)
        ac_src_B_l = ac_thr.partition_S(bB_l)
        ac_src_B_r = ac_thr.partition_S(bB_r)
        #LDS0
        ac_dest0_A_t = ac_thr.partition_D(lds0_A_t_wr)
        ac_dest0_A_b = ac_thr.partition_D(lds0_A_b_wr)
        ac_dest0_B_l = ac_thr.partition_D(lds0_B_l_wr)
        ac_dest0_B_r = ac_thr.partition_D(lds0_B_r_wr)
        #LDS1
        ac_dest1_A_t = ac_thr.partition_D(lds1_A_t_wr)
        ac_dest1_A_b = ac_thr.partition_D(lds1_A_b_wr)
        ac_dest1_B_l = ac_thr.partition_D(lds1_B_l_wr)
        ac_dest1_B_r = ac_thr.partition_D(lds1_B_r_wr)

        # tiled MMA, thread MMA
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        #tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (2, 1, 0)))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        # MMA copy A, B, C tiled copy
        s2r_tiled_copy_A = fx.make_tiled_copy_A(buffer_copy_atom_bf16, tiled_mma)
        s2r_tiled_copy_B = fx.make_tiled_copy_B(buffer_copy_atom_bf16, tiled_mma)
        # C tiled copy. make_tiled_copy_C is not used because C= B*A
        c_tile_mn = fx.make_tile(32, 32)
        # wave ((2, 2, 1), (1, 2, 0)):
        c_tv_layout =  fx.make_layout((((16, 4), 2, 2), 4), (((1, 128), 16, 512) , 32))   
        tiled_copy_C = fx.make_tiled_copy(buffer_copy_atom_f32, c_tv_layout, c_tile_mn)
        #MMA fragments
        #fragA layout:((a_val), m_rep, k_rep)
        #fragB layout:((b_val), n_rep, k_rep)
        #fragC layout:((c_val), m_rep, n_rep)

        #op1是A，op是B, fx.gemm(mma_atom, result, op1, op2, op3)的代码行为应该是：
        #C=A*B的情况下m_iter是m_rep, n_iter就是n_rep
        # m_iter = op1.shape[1]
        # n_iter = op2.shape[1]
        # k_iter = op1.shape[2] 
        # for m in range (m_iter):
        #     for n in range (n_iter):
        #         for k in range (k_iter):
        #             frag_C[None, m, n] += frag_A[None, m, k] * frag_B[None, k, n]

        #c=B*A, fx.gemm(mma_atom, C, B, A, C)
        #所以m_iter = n_rep, n_iter = m_rep, 
        #对frgaC的访问，frag_C[None, m_iter, n_iter]实际上是frag_C[None, n_rep, m_rep]
        frag_A_t = thr_mma.make_fragment_A(lds0_A_t_rd)
        frag_A_b = thr_mma.make_fragment_A(lds0_A_b_rd)
        frag_B_l = thr_mma.make_fragment_B(lds0_B_l_rd)
        frag_B_r = thr_mma.make_fragment_B(lds0_B_r_rd)
        #frag_C(val, m_rep, n_rep] -> frag_C[val, n_rep, m_rep]
        frag_C_tl = thr_mma.make_fragment_C(fx.select(bC_tl,[1,0]))
        frag_C_tr = thr_mma.make_fragment_C(fx.select(bC_tr,[1,0]))
        frag_C_bl = thr_mma.make_fragment_C(fx.select(bC_bl,[1,0]))
        frag_C_br = thr_mma.make_fragment_C(fx.select(bC_br,[1,0]))

        print(f'##frag_A_t={frag_A_t}')
        print(f'##frag_A_b={frag_A_b}')
        print(f'##frag_B_l={frag_B_l}')
        print(f'##frag_B_r={frag_B_r}')
        
        print(f'##frag_C_tl={frag_C_tl}')
        print(f'##frag_C_tr={frag_C_tr}')
        print(f'##frag_C_bl={frag_C_bl}')
        print(f'##frag_C_br={frag_C_br}')
        # from LDS to reigster partition
        ldsA_rd_thread = s2r_tiled_copy_A.get_slice(tid)
        ldsB_rd_thread = s2r_tiled_copy_B.get_slice(tid)
        s2r_src0_A_t = ldsA_rd_thread.partition_S(lds0_A_t_rd)
        s2r_src0_A_b = ldsA_rd_thread.partition_S(lds0_A_b_rd)
        s2r_src0_B_l = ldsB_rd_thread.partition_S(lds0_B_l_rd)
        s2r_src0_B_r = ldsB_rd_thread.partition_S(lds0_B_r_rd)
        
        s2r_src1_A_t = ldsA_rd_thread.partition_S(lds1_A_t_rd)
        s2r_src1_A_b = ldsA_rd_thread.partition_S(lds1_A_b_rd)
        s2r_src1_B_l = ldsB_rd_thread.partition_S(lds1_B_l_rd)
        s2r_src1_B_r = ldsB_rd_thread.partition_S(lds1_B_r_rd)
        ###MMA fragments retile to des
        dest_frag_A_t = ldsA_rd_thread.retile(frag_A_t)
        dest_frag_A_b = ldsA_rd_thread.retile(frag_A_b)
        dest_frag_B_l = ldsB_rd_thread.retile(frag_B_l)
        dest_frag_B_r = ldsB_rd_thread.retile(frag_B_r)

        frag_C_tl.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        frag_C_tr.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        frag_C_bl.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        frag_C_br.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
        acc_init = [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
        
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_B_l[None, None, None, 0], ac_dest0_B_l)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_A_t[None, None, None, 0], ac_dest0_A_t)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_A_b[None, None, None, 0], ac_dest0_A_b)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_B_r[None, None, None, 0], ac_dest0_B_r)
        rocdl.sched_barrier(0)

        fx.copy(async_copy_atom, ac_src_B_l[None, None, None, 1], ac_dest1_B_l)
        rocdl.sched_barrier(0)

        fx.copy(async_copy_atom, ac_src_A_t[None, None, None, 1], ac_dest1_A_t)
        rocdl.sched_barrier(0)

        fx.copy(async_copy_atom, ac_src_A_b[None, None, None, 1], ac_dest1_A_b)
        rocdl.sched_barrier(0)
        fx.copy(async_copy_atom, ac_src_B_r[None, None, None, 1], ac_dest1_B_r)
        rocdl.sched_barrier(0)
        
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=24))
        gpu.barrier()
        fx.copy(lsd_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
        fx.copy(lsd_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
        rocdl.sched_barrier(0)
        
        frag_C_tl.fill(0)
        frag_C_tr.fill(0)
        frag_C_bl.fill(0)
        frag_C_br.fill(0)
        rocdl.sched_barrier(0)

        wait_inline_asm = False
        mfma_inline_asm = True
        # for kidx, states in range(0, K // BLOCK_K - 2, 2, init=acc_init):    
        for kiter in const_expr.range(0, K // BLOCK_K - 2, 2):
            # frag_C_tl.store(states[0])
            # frag_C_tr.store(states[1])
            # frag_C_bl.store(states[2])
            # frag_C_br.store(states[3])
            # kiter = fx.Int32(kidx)
            mfma_16x16x64 = Mfma16x16x64(4, 4, mma_atom, use_inline_asm=mfma_inline_asm)

            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
            fx.copy(lsd_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
            fx.copy(async_copy_atom, ac_src_B_l[None, None, None, kiter+2], ac_dest0_B_l)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
            fx.copy(lsd_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
            fx.copy(async_copy_atom, ac_src_A_t[None, None, None, kiter+2], ac_dest0_A_t)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
            fx.copy(lsd_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
            fx.copy(async_copy_atom, ac_src_A_b[None, None, None, kiter+2], ac_dest0_A_b)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)
        
            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
            fx.copy(lsd_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
            fx.copy(async_copy_atom, ac_src_B_r[None, None, None, kiter+2], ac_dest0_B_r)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)

            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
            fx.copy(lsd_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
            fx.copy(async_copy_atom, ac_src_B_l[None, None, None, kiter+3], ac_dest1_B_l)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
            fx.copy(lsd_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
            fx.copy(async_copy_atom, ac_src_A_t[None, None, None, kiter+3], ac_dest1_A_t)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
            fx.copy(lsd_copy_atom, s2r_src0_B_l, dest_frag_B_l, pred=None)
            fx.copy(async_copy_atom, ac_src_A_b[None, None, None, kiter+3], ac_dest1_A_b)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)
            
            waitvmcnt_barrier(20)
            mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
            fx.copy(lsd_copy_atom, s2r_src0_A_t, dest_frag_A_t, pred=None)
            fx.copy(async_copy_atom, ac_src_B_r[None, None, None, kiter+3], ac_dest1_B_r)
            hot_loop_scheduler_mainloop()
            rocdl.sched_barrier(0)

            # results = yield [frag_C_tl.load(), frag_C_tr.load(), frag_C_bl.load(), frag_C_br.load()]
        #frag_C(val, n_rep, m_rep] -> frag_C[val, m_rep, n_rep]
        # frag_C_tl.store(results[0])
        # frag_C_tr.store(results[1])
        # frag_C_bl.store(results[2])
        # frag_C_br.store(results[3])

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=0))
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            
        mfma_16x16x64 = Mfma16x16x64(4, 4, mma_atom, use_inline_asm=mfma_inline_asm)
        gpu.barrier()
        mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
        fx.copy(lsd_copy_atom, s2r_src0_A_b, dest_frag_A_b, pred=None)
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        gpu.barrier()
        rocdl.sched_barrier(0)

        mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
        fx.copy(lsd_copy_atom, s2r_src0_B_r, dest_frag_B_r, pred=None)
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        gpu.barrier()
        rocdl.sched_barrier(0)

        mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
        fx.copy(lsd_copy_atom, s2r_src1_B_l, dest_frag_B_l, pred=None)
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        gpu.barrier()
        rocdl.sched_barrier(0)

        mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)
        fx.copy(lsd_copy_atom, s2r_src1_A_t, dest_frag_A_t, pred=None)
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        gpu.barrier()
        rocdl.sched_barrier(0)

        gpu.barrier()
        mfma_16x16x64.call_BxA(frag_A_t, frag_B_l, frag_C_tl)
        fx.copy(lsd_copy_atom, s2r_src1_A_b, dest_frag_A_b, pred=None)
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        gpu.barrier()
        rocdl.sched_barrier(0)

        mfma_16x16x64.call_BxA(frag_A_b, frag_B_l, frag_C_bl)
        fx.copy(lsd_copy_atom, s2r_src1_B_r, dest_frag_B_r, pred=None)
        rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
        gpu.barrier()
        rocdl.sched_barrier(0)

        mfma_16x16x64.call_BxA(frag_A_t, frag_B_r, frag_C_tr)
        mfma_16x16x64.call_BxA(frag_A_b, frag_B_r, frag_C_br)

        gpu.barrier()

        frag_C_tl = fx.select(frag_C_tl, [0, 2, 1])
        frag_C_tr = fx.select(frag_C_tr, [0, 2, 1])
        frag_C_bl = fx.select(frag_C_bl, [0, 2, 1])
        frag_C_br = fx.select(frag_C_br, [0, 2, 1])
        
        thr_copy_C = tiled_copy_C.get_slice(tid)
        dst_C_tl = thr_copy_C.partition_D(bC_tl)
        dst_C_tr = thr_copy_C.partition_D(bC_tr)
        dst_C_bl = thr_copy_C.partition_D(bC_bl)
        dst_C_br = thr_copy_C.partition_D(bC_br)
        
        src_frag_C_tl = thr_copy_C.retile(frag_C_tl)
        src_frag_C_tr = thr_copy_C.retile(frag_C_tr)
        src_frag_C_bl = thr_copy_C.retile(frag_C_bl)
        src_frag_C_br = thr_copy_C.retile(frag_C_br)
        fx.copy(buffer_copy_atom_f32, src_frag_C_tl, dst_C_tl, pred=None)
        fx.copy(buffer_copy_atom_f32, src_frag_C_tr, dst_C_tr, pred=None)
        fx.copy(buffer_copy_atom_f32, src_frag_C_bl, dst_C_bl, pred=None)
        fx.copy(buffer_copy_atom_f32, src_frag_C_br, dst_C_br, pred=None)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        M: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        
        value_attrs = {"rocdl.waves_per_eu": 1,
                    "passthrough": [["amdgpu-agpr-alloc", "256,256"],]
                    }
        # A_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
        # B_2d = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1))))
        # C_2d = fx.Tensor(fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1))))
        gemm_kernel(A, B, C, M, value_attrs=value_attrs,).launch(grid=(div_up(M, TILE_M)*div_up(N, TILE_N), 1, 1), block=(256, 1, 1), stream=stream)
        
    launch_gemm.compile_hints["llvm_options"] = {
        "amdgpu-mfma-vgpr-form": False,
    }
    return launch_gemm


TILE_M = 256
TILE_N = 256
TILE_K = 64
M = TILE_M *32
N = TILE_N*32
K = TILE_K*128
USE_SWIZZLE=_env_flag("SWIZZLE", "0")
enable_dump_ir(True)
# assert BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64, "BLOCK_M, BLOCK_N, BLOCK_K must be 128, 128, 64"
A = torch.randn(M, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
B = torch.randn(N, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
C = torch.zeros(M, N, dtype=torch.float32).cuda()
expected = A.to(torch.float32) @ B.to(torch.float32).T

hints = {
    "opt_level" : 2,
    "llvm_options": {"amdgpu-mfma-vgpr-form": False},
}
stream=torch.cuda.Stream()
launcher_gemm = compile_gemm(TILE_M = 256,
    TILE_N = 256,
    TILE_K = 64,
    N = N,
    K = K,
    dtype="bf16",
    pin_bf16_agpr=True,
    lds_swizzle=USE_SWIZZLE,
    pid_swizzle=False,
    permlane_epilogue=True)

compiled_gemm = flyc.compile[hints](launcher_gemm, A, B, C, stream)
compiled_gemm(A, B, C, M, stream)
torch.cuda.synchronize()

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
is_correct = torch.allclose(expected, C, atol=1e-5, rtol=1e-5)

print(f'{USE_SWIZZLE=} {is_correct=}')

import pyhip

def compare_perf(run_count=16, data_clones=32):
    _A = torch.randn(M, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
    _B = torch.randn(N, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
    _C = torch.zeros(M, N, dtype=torch.float32).cuda()

    _hints = {"opt_level": 2, "llvm_options": {"amdgpu-mfma-vgpr-form": False}}
    _stream = torch.cuda.current_stream()
    _compiled = flyc.compile[_hints](launcher_gemm, _A, _B, _C, _stream)

    # accuracy
    _expected = _A.to(torch.float32) @ _B.to(torch.float32).T
    _compiled(_A, _B, _C, M, _stream)
    torch.cuda.synchronize()
    acc = "pass" if torch.allclose(_expected, _C, atol=1e-5, rtol=1e-5) else "failed"

    As = [torch.randn(M, K, dtype=torch.bfloat16).cuda() for _ in range(data_clones)]
    Bs = [torch.randn(N, K, dtype=torch.bfloat16).cuda() for _ in range(data_clones)]
    Cs = [torch.zeros(M, N, dtype=torch.float32).cuda() for _ in range(data_clones)]

    flops = 2 * M * N * K
    mem_bytes = (M * K + N * K) * 2 + M * N * 4  # bf16 A+B + float32 C

    di = 0
    latencies = []
    torch_latencies = []

    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"gemm_{di}") as p:
            _compiled(As[di], Bs[di], Cs[di], M, _stream)
        latencies.append(p.dt_ms)

    for _ in range(run_count):
        di = (di + 1) % data_clones
        with pyhip.cudaPerf(flops, mem_bytes, name=f"torch_{di}") as p:
            _ = torch.nn.functional.linear(As[di], Bs[di]) 
        torch_latencies.append(p.dt_ms)

    latencies.sort()
    torch_latencies.sort()

    best_ms = latencies[0]
    tflops = flops / (best_ms * 1e-3) / 1e12
    bw_gbs = mem_bytes / (best_ms * 1e-3) / 1e9

    print(f"\n=== compare_perf  M={M} N={N} K={K} ===")
    print(f"acc:   {acc}")
    print(f"gemm:  {best_ms*1e3:.1f} us  {tflops:.2f} TFLOPS  {bw_gbs:.1f} GB/s")
    print(f"torch: {torch_latencies[0]*1e3:.1f} us")
    print(f"ratio: {torch_latencies[0]/best_ms:.2f}x")


compare_perf()