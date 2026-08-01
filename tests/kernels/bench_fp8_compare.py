"""Unified fp8 GEMM perf comparison: reference gemm_4wave_950 vs test_gemm_v9_fp8.

Both sides use the IDENTICAL cold-cache harness (32 rotated clones, best-of-N,
pyhip.cudaPerf). M=N=K=8192, TILE 256. Run on one GPU, gfx950.
"""
import importlib.util
import os
import sys

import torch
import flydsl.compiler as flyc
import pyhip

M = N = K = 8192
TILE_M = TILE_N = 256
TILE_K = 128  # fp8
DATA_CLONES = 32
RUN_COUNT = 50

FLOPS = 2 * M * N * K
MEM_BYTES = (M * K + N * K) * 1 + M * N * 2  # fp8 A+B (1B) + bf16 C (2B)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(kernel, arg_sets):
    # warmup: touch all clones (cold data rotation just like both original harnesses)
    for i in range(DATA_CLONES):
        kernel(*arg_sets[i])
    torch.cuda.synchronize()
    di = 0
    lat = []
    for _ in range(RUN_COUNT):
        di = (di + 1) % DATA_CLONES
        with pyhip.cudaPerf(FLOPS, MEM_BYTES, name=f"g{di}") as p:
            kernel(*arg_sets[di])
        lat.append(p.dt_ms)
    lat.sort()
    best_ms = lat[0]
    return best_ms, FLOPS / (best_ms * 1e-3) / 1e12, MEM_BYTES / (best_ms * 1e-3) / 1e9


def rand_fp8(shape):
    return torch.randint(-2, 3, shape, device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)


stream = torch.cuda.current_stream()
props = torch.cuda.get_device_properties()
assert "950" in props.gcnArchName, "fp8 MFMA_Scale requires gfx950"
torch.manual_seed(0)

# shared A / B raw data (same underlying values so both sides do the same math)
As = [rand_fp8((M, K)) for _ in range(DATA_CLONES)]
Bs = [rand_fp8((N, K)) for _ in range(DATA_CLONES)]
Cs_ref = [torch.empty((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(DATA_CLONES)]
Cs_mine = [torch.zeros((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(DATA_CLONES)]

results = {}

# ---------- reference: compile_gemm_950 num_waves=4 fp8 (preshuffle B + A padding) ----------
ref_mod = _load("/mywork/pyhip/tests/flydsl/test_gemm.py", "ref_gemm")
from aiter.ops.shuffle import shuffle_weight as aiter_shuffle  # reference's preshuffle

Bsh_ref = [aiter_shuffle(b).reshape(N // 16, -1) for b in Bs]
ref_args = [
    (Cs_ref[i].view(-1), As[i].view(torch.int8).view(-1), Bsh_ref[i].view(torch.int8).view(-1), M, stream)
    for i in range(DATA_CLONES)
]
ref_launcher = ref_mod.compile_gemm_950(TILE_M, TILE_N, N, K, 4, "fp8")
ref_kernel = flyc.compile[{"opt_level": 2}](ref_launcher, *ref_args[0])
results["ref_4wave_950_fp8"] = bench(ref_kernel, ref_args)

# ---------- mine: compile_gemm_fp8 (padding / swizzle / preshuffle) ----------
mine_mod = _load("/mywork/FlyDSL/tests/kernels/test_gemm_v9_fp8.py", "mine_gemm")
sys.path.insert(0, "/mywork/FlyDSL")
from tests.utils import shuffle_weight as my_shuffle  # layout (16,64) for fp8

configs = [
    ("mine_padding", dict(lds_swizzle=False, preshuffle_b=False)),
    ("mine_swizzle", dict(lds_swizzle=True, preshuffle_b=False)),
    ("mine_preshuffle", dict(lds_swizzle=False, preshuffle_b=True)),
]
for name, kw in configs:
    preshuffle = kw["preshuffle_b"]
    Bfeed = [my_shuffle(b, layout=(16, 64)) if preshuffle else b for b in Bs]
    my_args = [
        (As[i].view(torch.int8).view(-1), Bfeed[i].view(torch.int8).view(-1), Cs_mine[i].view(-1), M, stream)
        for i in range(DATA_CLONES)
    ]
    launcher = mine_mod.compile_gemm_fp8(TILE_M, TILE_N, TILE_K, N, K, **kw)
    kernel = flyc.compile[{"opt_level": 2}](launcher, *my_args[0])
    results[name] = bench(kernel, my_args)

print("\n================ fp8 GEMM  M=N=K=8192  TILE 256  (best-of-%d, %d cold clones) ================" % (RUN_COUNT, DATA_CLONES))
ref_tflops = results["ref_4wave_950_fp8"][1]
for name, (ms, tflops, gbs) in results.items():
    rel = tflops / ref_tflops
    print(f"{name:22s}: {ms*1e3:8.1f} us   {tflops:8.1f} TFLOPS   {gbs:7.1f} GB/s   ({rel:.3f}x ref)")
