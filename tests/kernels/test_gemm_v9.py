# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch
import math
import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.compiler as flyc
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T
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
    a0: fx.Array[BFloat16, BLOCK_M*BLOCK_K, 16]
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
    # B0 read layout
    B = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout(((8, BLOCK_N//8, N//BLOCK_N), K), ((BLOCK_N//8*K, K, BLOCK_N*K), 1))))
    
    bA = fx.flat_divide(A, (BLOCK_M, BLOCK_K))[None, None, bid_x, None]  # (BM, BK, k)
    bB = fx.flat_divide(B, (BLOCK_N, BLOCK_K))[None, None, bid_y, None]  # (BN, BK, k)
    bC = fx.flat_divide(C, (BLOCK_M, BLOCK_N))[None, None, bid_x, bid_y]  # (BM, BN)
    

    # read and write LDS tensor view.
    lds_layout_rd =fx.make_layout(((16, 8), (32, 2)), ((512+PADDING_ELEMS, 64), (1, 32)))
    lds_layout_wr =fx.make_layout(((8, 16), 64), ((64, 8*64+PADDING_ELEMS), 1))
    lds = fx.SharedAllocator().allocate(LDS_PADDING).peek()
    ldsB0_rd = fx.make_view(lds.b0.ptr, lds_layout_rd)
    ldsB0_wr = fx.make_view(lds.b0.ptr, lds_layout_wr)
    ldsA0_rd = fx.make_view(lds.a0.ptr, lds_layout_rd)
    ldsA0_wr = fx.make_view(lds.a0.ptr, lds_layout_wr)

    # copy atoms
    async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    lsd_copy_128b_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    buffer_copy_atom_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    buffer_copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    
    # MMA tiles, thread MMA
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    #tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (2, 1, 0)))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    # MMA copy A, B, C tiles.
    tiled_copy_A = fx.make_tiled_copy_A(buffer_copy_atom_bf16, tiled_mma)
    s2r_tiled_copy_B = fx.make_tiled_copy_B(buffer_copy_atom_bf16, tiled_mma)
    # C tile is B * A not A *B
    c_tile_mn = fx.make_tile(32, 32)
    # wave ((2, 2, 1), (2, 1, 0)):
    # c_tv_layout =  fx.make_layout((((16, 4), 2, 2), 4), (((1, 128), 512, 16) , 32))
    # wave ((2, 2, 1), (1, 2, 0)):
    c_tv_layout =  fx.make_layout((((16, 4), 2, 2), 4), (((1, 128), 16, 512) , 32))   
    tiled_copy_C = fx.make_tiled_copy(buffer_copy_atom_f32, c_tv_layout, c_tile_mn)
    
    # DMA copy tiles
    ac_tile_mn = fx.make_tile(32, 64)
    ac_tv_layout =  fx.make_layout(((8, 8, 4), 8), ((8*4*8, 1, 8), 4*8))
    ac_tiled_copy = fx.make_tiled_copy(buffer_copy_atom_bf16, ac_tv_layout, ac_tile_mn)
    ac_thr = ac_tiled_copy.get_slice(tid)
    
    # DMA copy B partition
    ac_B_src = ac_thr.partition_S(bB)
    ac_B_dest = ac_thr.partition_D(ldsB0_wr)

    # B from LDS to reigster partition
    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = s2r_tiled_copy_B.get_slice(tid)
    ldsB_rd_thread = s2r_tiled_copy_B.get_slice(tid)
    s2r_src = ldsB_rd_thread.partition_S(ldsB0_rd)
    
    # C tile is B * A not A *B
    c_tile_mn = fx.make_tile(32, 32)
    #wave ((2, 2, 1), (2, 1, 0)):
    # c_tv_layout =  fx.make_layout((((16, 4), 2, 2), 4), (((1, 128), 512, 16) , 32))
    #wave ((2, 2, 1), (1, 2, 0)):
    c_tv_layout =  fx.make_layout((((16, 4), 2, 2), 4), (((1, 128), 16, 512) , 32))   
    tiled_copy_C = fx.make_tiled_copy(buffer_copy_atom_f32, c_tv_layout, c_tile_mn)
    
    fx.utils.print_typst(tiled_copy_C, file="tiled_copy_C.typ")

    ###MMA fragments
    thr_copy_C = tiled_copy_C.get_slice(tid)
    frag_A = thr_mma.make_fragment_A(bA[None, None, 0])
    frag_B = thr_mma.make_fragment_B(ldsB0_rd)
    frag_C = thr_mma.make_fragment_C(fx.select(bC,[1,0]))

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_D(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = ldsB_rd_thread.retile(frag_B)
    frag_C.fill(0)

    for kiter in fx.range_constexpr(K // BLOCK_K):
        fx.copy(async_copy_atom, ac_B_src[None, None, None, kiter], ac_B_dest)
        gpu.barrier()
        fx.copy(buffer_copy_atom_bf16, copy_src_A[None, None, None, kiter], copy_frag_A, pred=None)
        fx.copy(lsd_copy_128b_atom, s2r_src, copy_frag_B, pred=None)
        # frag_C  = frag_B * frag_A
        fx.gemm(mma_atom, frag_C, frag_B, frag_A, frag_C)
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