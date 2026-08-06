# MUSA Level 1 上游算子源码清单

获取日期：2026-08-06

本目录保存此前检索到、与 KernelBench Level 1 相关的 MUSA C/C++ 上游源码。这里的“已有实现”分为三类，不能混为一谈：

1. 公开的 `.mu` / C++ 原生 kernel；
2. 公开的 PyTorch/MNN C++ 接入代码，底层调用闭源 muDNN 或 muBLAS；
3. 由多个基础算子组合得到的实现，并非针对 KernelBench 单题的融合 kernel。

目前仍未发现一个公开、完整、按 KernelBench Level 1 题号组织的 100 题 MUSA 高性能 kernel 合集。

## 固定版本

| 本地目录 | 上游 | 提交 |
| --- | --- | --- |
| `torch_musa/` | https://github.com/MooreThreads/torch_musa | `467bb873388939c646ac4525f463bcdcc3eb570e` |
| `mutlass/` | https://github.com/MooreThreads/mutlass | `995e03e0527f4897fc8dd18557865aef79737c81` |
| `mate/` | https://github.com/MooreThreads/mate | `e3e10c32545931024565efb80ef5712ed9d1e0c8` |
| `MNN/` | https://github.com/alibaba/MNN | `9059f0ab60b0ba1eeb35168fd46d261c1a4a435d`；仅稀疏获取 `source/backend/musa/` 及仓库根文件 |
| `MT-flashMLA/` | https://github.com/MooreThreads/MT-flashMLA | `c564db4a1b4a447c78973a2978a07b0fa6385843` |

## Level 1 对应关系

| Level 1 题号 | 类别/算子 | 主要源码来源 | 实现性质 |
| --- | --- | --- | --- |
| 1–4、6–10、13–18 | `matmul` / `mm` / `bmm` / 转置和三角矩阵乘法 | `torch_musa/.../ops/Matmul.cpp`、`ops/musa/Matmul.mu`，`mutlass/`，`mate/csrc/batch_gemm.mu` | 公开 kernel 与 muBLAS/MUTLASS 接入 |
| 5 | 矩阵标量乘 | `torch_musa` elementwise `mul.Scalar` | 通用逐元素实现 |
| 11 | 4D tensor matrix multiplication / `einsum` | 未注册 `einsum`，无题目级原生实现 | 缺口 |
| 12 | diagonal matrix multiplication / `diag` | 未注册 `diag`，无题目级原生实现 | 缺口 |
| 19–26、28–29、31–32 | 常见激活、Softmax、LogSoftmax | `torch_musa/.../ops/SoftMax.cpp` 及 elementwise kernels；`MNN/.../UnaryExecution.cu`、`SoftmaxExecution.cu` | 原生/库接入 |
| 25 | Swish | Sigmoid 与乘法 | 组合实现 |
| 27 | SELU | 未注册 `selu` | 缺口 |
| 30 | Softsign | 题目为 `x / (1 + |x|)`，由 `abs` 与逐元素运算组合 | 组合实现 |
| 33、35–36、40 | BatchNorm、GroupNorm、RMSNorm、LayerNorm | `torch_musa/.../ops/{BatchNorm,GroupNorm,RMSNorm,LayerNorm}.cpp`；MNN Norm 源码 | 多数为 muDNN 接入，RMSNorm 另有公开实现 |
| 34 | InstanceNorm | 未注册 `instance_norm` | 缺口 |
| 37–39 | Frobenius/L1/L2 Norm | `torch_musa` reduction/norm kernels | 原生或组合实现 |
| 41、44 | MaxPool1D、AvgPool1D | 未注册 `max_pool1d`/`avg_pool1d`；`Pool.cpp` 仅覆盖 2D/3D | 缺口 |
| 42–43、45–46 | 2D/3D Max/Avg Pool | `torch_musa/.../ops/Pool.cpp`；`MNN/.../PoolExecution.cu` | 主要为 muDNN 接入 |
| 47–49、51–53 | Sum/Mean/Max/Argmax/Argmin/Min | `torch_musa/.../ops/Reduce*.cpp`、`ops/musa/ReduceSumProdKernel.mu`；MNN Reduce/Arg 源码 | 公开 kernel 与通用 reduction |
| 50、54–87 | Conv、ConvTranspose、Depthwise、Grouped、Dilated | `torch_musa/.../ops/Conv.cpp`、`ops/musa/Conv.mu`；MNN Conv/Deconv 源码 | 公开接入代码，核心算法多由闭源 muDNN 提供 |
| 88 | MinGPT NewGELU | `tanh`、`pow` 等 elementwise kernels | 组合实现 |
| 89–90 | Cumsum、Cumprod | `torch_musa/.../ops/ScanKernels.cpp` | 原生扫描实现 |
| 91–93 | reverse/exclusive/masked cumsum | Scan 与 flip/cat/mask 等 | 组合实现 |
| 94、96 | MSELoss、HuberLoss | `torch_musa` loss/elementwise 路径 | 已有后端支持 |
| 95 | CrossEntropyLoss | `_fused_cross_entropy_loss_2d`、`cross_entropy_loss_2d_choice` 路径 | 已有后端支持，需按题目形状验证 |
| 97 | Scaled Dot Product Attention | `torch_musa/.../attention/mudnn/SDP.cpp`，`mutlass/experimental/fmha/*.mu`，MATE Flash Attention，`MT-flashMLA/csrc/*.mu` | 公开 FMHA/MLA 与 muDNN 接入；不同变体不保证直接适配题目 |
| 98 | KLDivLoss | 未注册 `kl_div` | 缺口 |
| 99 | TripletMarginLoss | 未注册 `triplet_margin` | 缺口 |
| 100 | HingeLoss | 题目为 `mean(clamp(1 - p*t, min=0))`，由 `clamp` 与 `mean` 组合 | 组合实现 |

## 重点源码入口

- `torch_musa/tools/ops_scanner/ops_list.md`
- `torch_musa/torch_musa/csrc/aten/ops/`
- `torch_musa/torch_musa/csrc/aten/ops/musa/`
- `torch_musa/torch_musa/csrc/aten/ops/attention/`
- `mutlass/examples/`、`mutlass/experimental/fmha/`
- `mate/csrc/`
- `MNN/source/backend/musa/execution/`
- `MT-flashMLA/csrc/`

## 使用注意

- 这些源码按上游许可证提供；复用前请检查各子目录的许可证。
- 单独复制某个 `.mu` 文件通常无法编译，MUTLASS、MATE 等实现依赖同仓库头文件、构建配置以及匹配的 MUSA SDK。
- MATE 和 MT-flashMLA 中部分实现面向 `mp_31`，当前 MTT S4000 是 `mp_22`，不可直接假定兼容。
- `torch_musa` 的算子清单表示后端注册/支持情况，不等价于每项都有公开的独立手写 kernel。
- 真正用于 KernelBench 前，应按题目 dtype、shape、stride、padding 和目标架构逐项编译与正确性验证。
- 逐题覆盖判定见 `level1_musa_coverage.csv` / `level1_musa_coverage.md`，由 `scripts/analyze_level1_musa_coverage.py` 重新生成。
