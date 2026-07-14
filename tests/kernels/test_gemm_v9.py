# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch
import math
import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.compiler as flyc
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T, Vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, vector

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
M = BLOCK_M *16
N = BLOCK_N*16
K = BLOCK_K*32
if 0:
    M = BLOCK_M
    N = BLOCK_N
    K = BLOCK_K * 4


# every 8 contineous row pad 16 elements. (need 128/8-1) * 16 elements padding totally.
PADDING_ELEMS = 16
PADDING_NUM = PADDING_ELEMS * (16 - 1)
@fx.struct
class LDS_PADDING:
    a0: fx.Array[BFloat16, BLOCK_M*BLOCK_K+PADDING_NUM, 16]
    b0: fx.Array[BFloat16, BLOCK_N*BLOCK_K+PADDING_NUM, 16]

@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid_x, bid_y, _ = fx.block_idx

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)
    # A, B read layout
    B = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout(((8, BLOCK_N//8, N//BLOCK_N), K), ((BLOCK_N//8*K, K, BLOCK_N*K), 1))))
    A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(((8, BLOCK_M//8, M//BLOCK_M), K), ((BLOCK_M//8*K, K, BLOCK_M*K), 1))))

    bA = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x, None]  # (BM, BK, k)
    bB = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y, None]  # (BN, BK, k)
    bC = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x, bid_y]  # (BM, BN)
    

    # read and write LDS tensor view.
    lds_layout_rd =fx.make_layout(((16, 8), (32, 2)), ((512+PADDING_ELEMS, 64), (1, 32)))
    lds_layout_wr =fx.make_layout(((8, 16), 64), ((64, 8*64+PADDING_ELEMS), 1))
    lds = fx.SharedAllocator().allocate(LDS_PADDING).peek()

    ldsA0_rd = fx.make_view(lds.a0.ptr, lds_layout_rd)
    ldsA0_wr = fx.make_view(lds.a0.ptr, lds_layout_wr)

    ldsB0_rd = fx.make_view(lds.b0.ptr, lds_layout_rd)
    ldsB0_wr = fx.make_view(lds.b0.ptr, lds_layout_wr)

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
    ac_B_src = ac_thr.partition_S(bB)
    ac_B_dest = ac_thr.partition_D(ldsB0_wr)
    ac_A_src = ac_thr.partition_S(bA)
    ac_A_dest = ac_thr.partition_D(ldsA0_wr)
    
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
    frag_A = thr_mma.make_fragment_A(ldsA0_rd)
    frag_B = thr_mma.make_fragment_B(ldsB0_rd)
    #frag_C(val, m_rep, n_rep] -> frag_C[val, n_rep, m_rep]
    frag_C = thr_mma.make_fragment_C(fx.select(bC,[1,0]))
    
    # B from LDS to reigster partition
    ldsA_rd_thread = s2r_tiled_copy_A.get_slice(tid)
    ldsB_rd_thread = s2r_tiled_copy_B.get_slice(tid)
    s2r_B0_src = ldsB_rd_thread.partition_S(ldsB0_rd)
    s2r_A0_src = ldsA_rd_thread.partition_S(ldsA0_rd)

    ###MMA fragments
    thr_copy_C = tiled_copy_C.get_slice(tid)
    copy_dst_C = thr_copy_C.partition_D(bC)

    copy_frag_A = ldsA_rd_thread.retile(frag_A)
    copy_frag_B = ldsB_rd_thread.retile(frag_B)

    frag_C.store(Vector.filled(BLOCK_M * BLOCK_N // 64 // 4, 0, fx.Float32))
    acc_init = [frag_C.load()]
        
    for kidx, states in range(0, K // BLOCK_K - 0, 1, init=acc_init):    
    # for kiter in fx.range_constexpr(K // BLOCK_K):
        frag_C.store(states[0])
        kiter = fx.Int32(kidx)
        gpu.barrier()
        fx.copy(async_copy_atom, ac_B_src[None, None, None, kiter], ac_B_dest)
        fx.copy(async_copy_atom, ac_A_src[None, None, None, kiter], ac_A_dest)
        gpu.barrier()
        fx.copy(lsd_copy_atom, s2r_A0_src, copy_frag_A, pred=None)
        fx.copy(lsd_copy_atom, s2r_B0_src, copy_frag_B, pred=None)
        # frag_C  = frag_B * frag_A
        fx.gemm(mma_atom, frag_C, frag_B, frag_A, frag_C)
        results = yield [frag_C.load()]
    #frag_C(val, n_rep, m_rep] -> frag_C[val, m_rep, n_rep]
    frag_C.store(results)
    frag_C = fx.select(frag_C, [0, 2, 1])
    copy_frag_C = thr_copy_C.retile(frag_C)
    fx.copy(buffer_copy_atom_f32, copy_frag_C, copy_dst_C, pred=None)



@flyc.jit
def tiledMma(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    
    gemm_kernel(A, B, C).launch(grid=(M // BLOCK_M, N // BLOCK_N, 1), block=(256, 1, 1), stream=stream)

assert BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64, "BLOCK_M, BLOCK_N, BLOCK_K must be 128, 128, 64"
A = torch.randn(M, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
B = torch.randn(N, K, dtype=torch.bfloat16).cuda() / math.sqrt(K)
C = torch.zeros(M, N, dtype=torch.float32).cuda()
expected = A.to(torch.float32) @ B.to(torch.float32).T
tiledMma(A, B, C, stream=torch.cuda.Stream())

torch.cuda.synchronize()

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
is_correct = torch.allclose(expected, C, atol=1e-5, rtol=1e-5)

print("Result correct:", is_correct)
# if not is_correct:
#     m_tiles = BLOCK_M // 32
#     n_tiles = BLOCK_N // 32
#     for i in range(m_tiles):
#         base_m = i * 32
#         for j in range(n_tiles):
#             base_n = j * 32
#             C00 = C[base_m:base_m+16, base_n:base_n+16]
#             C01 = C[base_m:base_m+16, base_n+16:base_n+32]
#             C10 = C[base_m+16:base_m+32, base_n:base_n+16]
#             C11 = C[base_m+16:base_m+32, base_n+16:base_n+32]

#             expected00 = expected[base_m:base_m+16, base_n:base_n+16]
#             expected01 = expected[base_m:base_m+16, base_n+16:base_n+32]
#             expected10 = expected[base_m+16:base_m+32, base_n:base_n+16]
#             expected11 = expected[base_m+16:base_m+32, base_n+16:base_n+32]
            
#             print(f'#################base_m={base_m}, base_n={base_n}#####################')
#             print(f'{C00=}\n, {expected00=}')
#             print(f'{C01=}\n, {expected01=}')
#             print(f'{C10=}\n, {expected10=}')
#             print(f'{C11=}\n, {expected11=}')