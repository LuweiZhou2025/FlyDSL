# test_gemm_v9 Epilogue / Swizzle Sweep

问题规模：`M = N = K = 8192`，`TILE_M = TILE_N = 256`，`TILE_K = 64`，dtype = bf16（输出也是 bf16）。
每个配置取 15 次运行的最优延迟。精度校验 `atol=0.03, rtol=0.01`。

## 结果（8 组）

| permlane_epilogue | lds_swizzle | pid_swizzle | acc | 延迟 (us) | TFLOPS |
|:---:|:---:|:---:|:---:|---:|---:|
| False | False | False | OK | 750.7 | 1464.7 |
| False | False | True  | OK | 746.0 | 1473.9 |
| False | True  | False | OK | 807.6 | 1361.4 |
| False | True  | True  | OK | 803.1 | 1369.0 |
| True  | False | False | OK | **739.6** | **1486.5** |
| True  | False | True  | OK | 743.2 | 1479.3 |
| True  | True  | False | OK | 799.8 | 1374.8 |
| True  | True  | True  | OK | 839.1 | 1310.4 |

参考：torch `F.linear` 约 715–721 us。

## 各因素影响

- **lds_swizzle（影响最大，负面）**：打开后稳定回退约 7–13%。当前 padding 版（关闭 swizzle）已用「每 8 行 pad 16 元素」消除 LDS bank conflict，swizzle 走的是 `make_ordered_layout` 直连、reader/writer 布局未优化，因此更慢。建议保持关闭。
- **permlane_epilogue（正面，小幅）**：lds_swizzle=False 时，permlane 一次 128-bit 写 8×bf16 比简单 tiled-copy 快约 1–1.5%（750.7→739.6、746.0→743.2）。收益小是因为 epilogue 已与最后的 MFMA 交织掩盖，store 非瓶颈。
- **pid_swizzle（影响最小，接近噪声）**：XCD remapping 在方形规整的 8192³ 上收益 <1%，数据里正负都有。主要在非方形 / L2 局部性敏感场景才体现。

## 最优组合

**permlane_epilogue=True, lds_swizzle=False, pid_swizzle=False → 739.6 us / 1486.5 TFLOPS（约 0.98x torch）。**
