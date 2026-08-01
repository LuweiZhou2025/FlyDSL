"""8-wave fp8 GEMM comparison: my tile-based gemm_fp8_8wave vs reference gemm_8wave_950.

Identical cold-cache harness (32 rotated clones, best-of-N, pyhip.cudaPerf).
M=N=K from env (default 8192), TILE 256. gfx950, one GPU.
"""
import importlib.util
import os
import sys

import torch
import flydsl.compiler as flyc
import pyhip

M = int(os.environ.get("GEMM_M", 8192))
N = int(os.environ.get("GEMM_N", 8192))
K = int(os.environ.get("GEMM_K", 8192))
TILE_M = TILE_N = 256
TILE_K = 128
DATA_CLONES = 32
RUN_COUNT = 50
FLOPS = 2 * M * N * K
MEM_BYTES = (M * K + N * K) * 1 + M * N * 2


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(kernel, arg_sets):
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
    return lat[0], FLOPS / (lat[0] * 1e-3) / 1e12, MEM_BYTES / (lat[0] * 1e-3) / 1e9


def rand_fp8(shape):
    return torch.randint(-2, 3, shape, device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)


stream = torch.cuda.current_stream()
assert "950" in torch.cuda.get_device_properties().gcnArchName
torch.manual_seed(0)

As = [rand_fp8((M, K)) for _ in range(DATA_CLONES)]
Bs = [rand_fp8((N, K)) for _ in range(DATA_CLONES)]
Cs_ref = [torch.empty((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(DATA_CLONES)]
Cs_mine = [torch.zeros((M, N), device="cuda", dtype=torch.bfloat16) for _ in range(DATA_CLONES)]
results = {}

# reference gemm_8wave_950 fp8 (preshuffle B via aiter shuffle_weight)
ref_mod = _load("/mywork/pyhip/tests/flydsl/test_gemm.py", "ref_gemm")
from aiter.ops.shuffle import shuffle_weight as aiter_shuffle

Bsh_ref = [aiter_shuffle(b).reshape(N // 16, -1) for b in Bs]
ref_args = [
    (Cs_ref[i].view(-1), As[i].view(torch.int8).view(-1), Bsh_ref[i].view(torch.int8).view(-1), M, stream)
    for i in range(DATA_CLONES)
]
ref_launcher = ref_mod.compile_gemm_950(TILE_M, TILE_N, N, K, 8, "fp8")
ref_kernel = flyc.compile[{"opt_level": 2}](ref_launcher, *ref_args[0])
results["ref_8wave_950_fp8"] = bench(ref_kernel, ref_args)

# mine: gemm_fp8_8wave (preshuffle B, matching reference)
mine_mod = _load("/mywork/FlyDSL/tests/kernels/test_gemm_v9_fp8_8wave.py", "mine_8wave")
sys.path.insert(0, "/mywork/FlyDSL")
from tests.utils import shuffle_weight as my_shuffle

Bfeed = [my_shuffle(b, layout=(16, 64)) for b in Bs]
my_args = [
    (As[i].view(torch.int8).view(-1), Bfeed[i].view(torch.int8).view(-1), Cs_mine[i].view(-1), M, stream)
    for i in range(DATA_CLONES)
]
launcher = mine_mod.compile_gemm_fp8_8wave(TILE_M, TILE_N, TILE_K, N, K, permlane_epilogue=True, preshuffle_b=True)
kernel = flyc.compile[{"opt_level": 2}](launcher, *my_args[0])
results["mine_8wave_pre"] = bench(kernel, my_args)

print("\n========= fp8 8-wave  M=N=K=%d  TILE 256  (best-of-%d, %d cold clones) =========" % (M, RUN_COUNT, DATA_CLONES))
ref_tflops = results["ref_8wave_950_fp8"][1]
for name, (ms, tflops, gbs) in results.items():
    print(f"{name:22s}: {ms*1e3:8.1f} us   {tflops:8.1f} TFLOPS   {gbs:7.1f} GB/s   ({tflops/ref_tflops:.3f}x ref)")
