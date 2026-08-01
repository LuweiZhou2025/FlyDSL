"""Run ONE fp8 kernel repeatedly for rocprof counter capture. argv[1] in {ref, preshuffle, padding}."""
import importlib.util
import sys

import torch
import flydsl.compiler as flyc

M = N = K = 8192
TILE_M = TILE_N = 256
TILE_K = 128
which = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rand_fp8(shape):
    return torch.randint(-2, 3, shape, device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)


stream = torch.cuda.current_stream()
torch.manual_seed(0)
a = rand_fp8((M, K))
b = rand_fp8((N, K))
c = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)

if which == "ref":
    ref_mod = _load("/mywork/pyhip/tests/flydsl/test_gemm.py", "ref_gemm")
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle
    bsh = aiter_shuffle(b).reshape(N // 16, -1)
    args = (c.view(-1), a.view(torch.int8).view(-1), bsh.view(torch.int8).view(-1), M, stream)
    launcher = ref_mod.compile_gemm_950(TILE_M, TILE_N, N, K, 4, "fp8")
else:
    mine_mod = _load("/mywork/FlyDSL/tests/kernels/test_gemm_v9_fp8.py", "mine_gemm")
    sys.path.insert(0, "/mywork/FlyDSL")
    from tests.utils import shuffle_weight as my_shuffle
    kw = {"padding": dict(lds_swizzle=False, preshuffle_b=False),
          "swizzle": dict(lds_swizzle=True, preshuffle_b=False),
          "preshuffle": dict(lds_swizzle=False, preshuffle_b=True)}[which]
    bfeed = my_shuffle(b, layout=(16, 64)) if kw["preshuffle_b"] else b
    args = (a.view(torch.int8).view(-1), bfeed.view(torch.int8).view(-1), c.view(-1), M, stream)
    launcher = mine_mod.compile_gemm_fp8(TILE_M, TILE_N, TILE_K, N, K, **kw)

kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
kernel(*args)
torch.cuda.synchronize()
for _ in range(ITERS):
    kernel(*args)
torch.cuda.synchronize()
