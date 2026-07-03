# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl, vector
NUM_CU = 1
N_ITER = 4
M_REPEAT = 4
WAVE_NUM=4
M_BLOCK = WAVE_NUM * 8 * M_REPEAT
N_BLOCK= 64

M = NUM_CU * M_BLOCK
N = N_BLOCK * N_ITER
def enable_dump_ir(enable_debug_info=True):
    if enable_debug_info:
        import os
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


LDS_SWIZZLE=False

@fx.struct
class SharedStorage:
    a0: fx.Array[BFloat16, M_BLOCK*N_BLOCK, 16]
    # a1: fx.Array[BFloat16, M_BLOCK*N_BLOC, 16]


       
@flyc.kernel
def copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x


    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()   
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0))

    sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), LDS_layout_A)

    bA = fx.flat_divide(A, (M_BLOCK, N_BLOCK))
    bB = fx.flat_divide(B, (M_BLOCK, N_BLOCK))
    # print(f'###{bA=} \n {bB=}')
    bA = bA[None, None, bid, None]
    bB = bB[None, None, bid, None]

    thr_layout = fx.make_layout((8*WAVE_NUM , 8), (8, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    # thr_layout = fx.make_layout((16, 4 * WAVE_NUM), (1, 16))
    # val_layout = fx.make_layout((4, 1), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)

    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    g2s_src = thr_copy.partition_S(bA)
    g2s_dest = thr_copy.partition_D(sA)
    g2s_frag = fx.make_fragment_like(g2s_dest)
    # print(f'\n####\n{g2s_src=} \n {g2s_dest=} \n {g2s_frag=}\n')

    
    s2g_src = thr_copy.partition_S(sA)
    s2g_frag = fx.make_fragment_like(s2g_src)

    # print(f"\n####\n{partition_src=} \n")
    s2g_dst = thr_copy.partition_D(bB)

    for block_idx in range(0, N_ITER):
        fx.copy(copy_atom, g2s_src[None, None, None, block_idx], g2s_frag)
        fx.copy(uni_copy_128b, g2s_frag, g2s_dest)
        fx.copy(uni_copy_128b, s2g_src, s2g_frag)
        fx.copy(copy_atom, s2g_frag, s2g_dst[None, None, None, block_idx])

@flyc.kernel
def async_copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    B = fx.rocdl.make_buffer_tensor(B)
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()   
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0))

    # if LDS_SWIZZLE:
    #     LDS_layout_A = fx.make_composed_layout(
    #         fx.static(fx.SwizzleType.get(3, 3, 3)),
    #         fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0)),
    #     )
    sA = fx.make_view(lds.a0.ptr, LDS_layout_A)
    bB = fx.flat_divide(B, (M_BLOCK, N_BLOCK))
    bB = bB[None, None, bid, None]


    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    thr_layout = fx.make_layout((WAVE_NUM*8 , 8), (8, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))

    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)
    s2g_src = thr_copy.partition_S(sA)
    s2g_frag = fx.make_fragment_like(s2g_src)
    s2g_dst = thr_copy.partition_D(bB)
    


    elem_bytes = 2
    dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    gA_flat = fx.rocdl.make_buffer_tensor(
        (fx.make_view(fx.get_iter(A), fx.make_layout(65536 * N, 1))),
        max_size=False,
        num_records_bytes=fx.Int64(M) * fx.Int64(N) * fx.Int64(elem_bytes),
    )

    gA_div = fx.logical_divide(gA_flat, fx.make_layout(1, 1))
    print(f'#####{gA_div=}\n{gA_flat=}')
    sA_i8_ptr = fx.recast_iter(Int8, lds.a0.ptr)
    sA_ptr = lds.a0.ptr


    bx_m = bid * M_BLOCK
    wave_id = tid // 64
    # 256*8. 256 threads load contineously
    load_bytes = 16
    # each wave load 64*16 btyes.
    wave_stride_bytes = 64 * load_bytes
    # num_a_loads = M_BLOCK // (WAVE_NUM*8)
    def dma_a_to_lds(blk_idx):
        wave_off = fx.rocdl.readfirstlane(fx.Int32.ir_type, wave_id * wave_stride_bytes)
        lds_ptr = fx.add_offset(sA_i8_ptr, wave_off)
        # lds_ptr = fx.add_offset(sA_ptr, wave_off//2)
        base_n = blk_idx * N_BLOCK
        total_threads = WAVE_NUM * 64
        step_bytes = total_threads * load_bytes
        for i in range_constexpr(M_REPEAT):
            pos_bytes =  i * total_threads * load_bytes + tid * load_bytes
            elem_idx = pos_bytes // elem_bytes
            m = elem_idx // N_BLOCK
            n = elem_idx % N_BLOCK
            n_swz = n
            # if LDS_SWIZZLE:
                # n_swz = n ^ ((m % k_blocks16_dma) * elems_per_16b)
                # k_swz = ((k // elems_per_16b) ^ (m % k_blocks16_dma)) * elems_per_16b
            # print(f'###############################################')
            offset = (m * N + base_n + n_swz)
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            src = fx.slice(gA_div, (None, fx.Int32(offset)))
            fx.copy(dma_atom, src, dst)
            lds_ptr = fx.add_offset(lds_ptr, step_bytes)

    for block_idx in range(0, N_ITER):
        dma_a_to_lds(block_idx)
        # rocdl.s_waitcnt(0)
        # gpu.barrier()
        fx.copy(uni_copy_128b, s2g_src, s2g_frag)
        fx.copy(copy_atom, s2g_frag, s2g_dst[None, None, None, block_idx])
 

@flyc.jit
def tiledCopy(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    M_max = 65536
    arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M_max, N), (N, 1))))
    async_copy_kernel(arg_a_2d, B).launch(grid=(NUM_CU, 1, 1), block=(WAVE_NUM*64, 1, 1), smem=M_BLOCK*N_BLOCK*2, stream=stream)


enable_dump_ir(True)
A = torch.arange(M * N, dtype=torch.bfloat16).reshape(M, N).cuda()
B = torch.zeros(M, N, dtype=torch.bfloat16).cuda()


tiledCopy(A, B, stream=torch.cuda.Stream())

torch.cuda.synchronize()


# print(A[0])
# print(B[0])

is_correct = torch.allclose(A, B)
print("Result correct:", is_correct)
if not is_correct:
    print("A:", A)
    print("B:", B)