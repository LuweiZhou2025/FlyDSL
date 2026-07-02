# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

NUM_CU = 1
N_ITER = 4
M_BLOCK = 128
N_BLOCK = 64

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


@flyc.kernel
def copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x


    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)


    LDS_layout_A = fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 3, 3)),
        fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0)),
    )
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0))
    sA = fx.make_view(fx.get_dyn_shared(fx.BFloat16), LDS_layout_A)
    bA = fx.flat_divide(A, (M_BLOCK, N_BLOCK))
    bB = fx.flat_divide(B, (M_BLOCK, N_BLOCK))
    print(f'###{bA=} \n {bB=}')
    bA = bA[None, None, bid, None]
    bB = bB[None, None, bid, None]

    thr_layout = fx.make_layout((32 , 8), (8, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    # thr_layout = fx.make_layout((16, 4 * NUM_WAVE), (1, 16))
    # val_layout = fx.make_layout((4, 1), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    uni_copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)

    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    print(f'###{tile_mn=}')

    # fx.utils.print_typst(tiled_copy, file="tiled_copy.typ")
    # print(f"{tile_mn=}")
    # print(f"{tv_layout=}")
    # print(f"\n####\n{tiled_copy=}")
    thr_copy = tiled_copy.get_slice(tid)
    # print(f"\n####\n{thr_copy=}")

    g2s_src = thr_copy.partition_S(bA)
    g2s_dest = thr_copy.partition_D(sA)
    g2s_frag = fx.make_fragment_like(g2s_dest)
    print(f'\n####\n{g2s_src=} \n {g2s_dest=} \n {g2s_frag=}\n')

    
    s2g_src = thr_copy.partition_S(sA)
    s2g_frag = fx.make_fragment_like(s2g_src)

    # print(f"\n####\n{partition_src=} \n")
    s2g_dst = thr_copy.partition_D(bB)

    for k_iter in range(0, N_ITER):
        fx.copy(copy_atom, g2s_src[None, None, None, k_iter], g2s_frag)
        fx.copy(uni_copy_128b, g2s_frag, g2s_dest)
        fx.copy(uni_copy_128b, s2g_src, s2g_frag)
        fx.copy(copy_atom, s2g_frag, s2g_dst[None, None, None, k_iter])

@flyc.jit
def tiledCopy(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    copy_kernel(A, B).launch(grid=(NUM_CU, 1, 1), block=(256, 1, 1), smem=32*64*2, stream=stream)


enable_dump_ir(False)
M, N = NUM_CU * M_BLOCK, N_BLOCK * N_ITER
A = torch.arange(M * N, dtype=torch.bfloat16).reshape(M, N).cuda()
B = torch.zeros(M, N, dtype=torch.bfloat16).cuda()


tiledCopy(A, B, stream=torch.cuda.Stream())

torch.cuda.synchronize()

is_correct = torch.allclose(A, B)
print("Result correct:", is_correct)
if not is_correct:
    print("A:", A)
    print("B:", B)