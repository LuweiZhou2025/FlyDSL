"""Profile LDS bank conflicts + wait cycles for one 8-wave fp8 kernel. argv[1] in {ref,mine}."""
import importlib.util
import sys
import torch
import flydsl.compiler as flyc

M = N = K = 8192
TILE_M = TILE_N = 256
TILE_K = 128
which = sys.argv[1]


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
    ref = _load("/mywork/pyhip/tests/flydsl/test_gemm.py", "ref_gemm")
    from aiter.ops.shuffle import shuffle_weight as sh
    bsh = sh(b).reshape(N // 16, -1)
    args = (c.view(-1), a.view(torch.int8).view(-1), bsh.view(torch.int8).view(-1), M, stream)
    launcher = ref.compile_gemm_950(TILE_M, TILE_N, N, K, 8, "fp8")
else:
    mine = _load("/mywork/FlyDSL/tests/kernels/test_gemm_v9_fp8_8wave.py", "mine_8wave")
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), M, stream)
    launcher = mine.compile_gemm_fp8_8wave(TILE_M, TILE_N, TILE_K, N, K, permlane_epilogue=True)

kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
for _ in range(3):
    kernel(*args)
torch.cuda.synchronize()
