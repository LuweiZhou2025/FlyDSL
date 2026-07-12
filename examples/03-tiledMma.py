# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

block_m = 16
block_n = 16
block_k = 32


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.zipped_divide(A, (block_m, block_k)) 
    bB = fx.zipped_divide(B, (block_n, block_k)) 
    bC = fx.zipped_divide(C, (block_m, block_n))

    bA = fx.slice(bA, (None, bid))# (BM, BK, k)
    bB = fx.slice(bB, (None, bid))# (BN, BK, k)
    bC = fx.slice(bC, (None, bid))# (BM, BN

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    
    copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    

    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    # tiled_copy_C = fx.make_tiled_copy_C(copy_atom_f32, tiled_mma)
    
    c_tile_mn = fx.make_tile(16, 16)
    c_tv_layout =  fx.make_layout(((16, 4), 4), ((1, 64) , 16))    
    tiled_copy_C = fx.make_tiled_copy(copy_atom_f32, c_tv_layout, c_tile_mn)
    
    print(f'{tiled_copy_A=}')
    print(f'{tiled_copy_B=}')
    print(f'{tiled_copy_C=}')

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    print(f"{copy_src_A=}")
    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)

    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_B, frag_A, frag_C)

    fx.copy(copy_atom_f32, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def tiledMma(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


M, N, K = block_m, block_n, block_k
A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
C = torch.zeros(M, N, dtype=torch.float32).cuda()

tiledMma(A, B, C, stream=torch.cuda.Stream())

torch.cuda.synchronize()

expected = A.to(torch.float32) @ B.to(torch.float32).T
is_correct = torch.allclose(C, expected, atol=1e-5, rtol=1e-5)
print("Result correct:", is_correct)
if not is_correct:
    print("Max diff:", (C - expected).abs().max().item())
    print("Expected:", expected)
    print("Got:", C)
