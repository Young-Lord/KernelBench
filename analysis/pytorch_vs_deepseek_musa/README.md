# 原生 PyTorch 与 DeepSeek MUSA 算子性能差异分析

## 1. 结论摘要

100 道 Level 1 题按 `speedup = PyTorch eager / DeepSeek` 分类：

| 类别 | 区间 | 题数 |
|---|---:|---:|
| 强加速 | `speedup >= 1.1` | 6 |
| 近似持平 | `0.9 <= speedup < 1.1` | 3 |
| 中度慢化 | `0.1 <= speedup < 0.9` | 47 |
| 严重慢化 | `speedup < 0.1` | 40 |
| 失败或无法计时 | 无有效 speedup | 4 |

本次从不同区间选择 7 个代表题并采集动态 kernel 时间线。结果表明：

1. DeepSeek 真正占优的主要场景是把 PyTorch 逐算子表达式融合为一次遍历，减少中间张量和全局内存流量。
2. 对 PyTorch 已进入 MUSA 汇编级 GEMM/卷积 kernel 的算子，DeepSeek 的逐输出朴素实现通常严重落后，原因是缺少 shared-memory/register tiling、数据复用和专用微内核。
3. kernel launch 更少不等于更快。Problem 97 将 5 次优化 kernel 融合为 1 次朴素 attention kernel，却慢约 452 倍。
4. 近似持平的算子通常具有相同的数据遍历下界；此时差异来自索引、循环展开、内部任务拆分等常数项。
5. 4 个无有效速度的题中，3 个是候选 `forward()` 原地销毁评测输入，1 个是后端正确性障碍，不能解释为性能结果。

## 2. 官方 profiling 工具

摩尔线程官方工具套件为 Moore Perf Tools：

- **Moore Perf System**：系统级时间线、MUSA API trace、CPU/GPU 并行关系、统计和专家分析；CLI 为 `msys`。
- **Moore Perf Compute**：单 kernel 硬件指标、Roofline、LaunchStats、MemoryWorkloadAnalysis、SpeedOfLight；CLI 为 `mcu`。
- **MUSA Compute Sanitizer**：越界访问、内存泄漏与 API 错误定位；CLI 为 `mt-compute-sanitizer`。

官方资料：

- <https://docs.mthreads.com/mooreperf/mooreperf-doc-online/introduction/>
- <https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_system/user_guide/>
- <https://docs.mthreads.com/en/mooreperf/mooreperf-doc-online/moore_perf_compute/user_guide/>
- <https://docs.mthreads.com/en/mooreperf/mooreperf-doc-online/moore_perf_compute/quickstart/>

本服务器检查结果：

```text
GPU                 MTT S4000, 48 GiB
Driver              2.7.0
PyTorch             2.2.0
torch_musa          1.3.0
msys                未安装
mcu                 未安装
mt-compute-sanitizer 未安装
```

官方当前 Moore Perf Compute 文档列出的 S4000 前置环境为 Driver 5.2.0 和 MUSA SDK 5.2.0。本机驱动明显更旧，因此没有直接安装新版工具，避免破坏已经可运行的 KernelBench 环境。升级驱动和 SDK 后建议使用：

```bash
msys profile --trace=musa -o problem_88_system \
  /root/miniconda3/envs/kernelbench-musa/bin/python <单题脚本>

mcu -k 'new_gelu_kernel' -o problem_88_compute.mcu-rep \
  /root/miniconda3/envs/kernelbench-musa/bin/python <单题脚本>
```

## 3. 当前采用的分析方法

在不能使用 `msys/mcu` 的当前环境中，采用以下证据链：

1. 已归档 baseline：5 次正确性、100 次 MUSA event 计时。
2. torch_musa Kineto：`ProfilerActivity.MUSA`，获取真实 kernel 名称、调用次数和 device time。
3. 源码分析：数学工作量、grid/block 映射、串行循环、访存连续性和数据复用。
4. 控制变量实验：Problem 88 比较逐算子表达式与 PyTorch fused GELU；Problem 47 使用已有指针步进消融结果。

`profile_operator_pair.py` 会预热两种实现，各采集一个 forward，并导出 Chrome trace 和 JSON 摘要。注意 profiler 中 ATen 父事件会重复包含底层 kernel 时间；汇总只统计底层设备 kernel。

## 4. 代表题选择

| Problem | 类别 | 选择理由 |
|---:|---|---|
| 88 MinGPTNewGelu | 强加速 | 典型逐元素表达式融合，可做等价 fused PyTorch 消融 |
| 45 Average Pooling 2D | 近似持平 | 大规模池化，两边接近一次遍历下界 |
| 47 Sum reduction | 近似持平 | 两边均为单 kernel，可隔离 kernel 内索引开销 |
| 6 large-K matmul | 中度慢化 | 与 Problem 1 对照，说明形状如何改变朴素 GEMM 的损失 |
| 1 square matmul | 严重慢化 | 原生汇编 SGEMM 对比无 tiling 的逐输出点积 |
| 55 Conv2d | 严重慢化 | 原生汇编卷积对比直接卷积，launch 数相同 |
| 97 SDPA | 极端慢化 | 证明“融合/更少 launch”不一定更快 |

## 5. 动态 profiling 结果

正式时间来自 100 次 baseline；profile 时间来自单个已预热 forward，仅用于识别 kernel 结构。两者数值方向一致。

| Problem | 正式 PyTorch | 正式 DeepSeek | speedup | PyTorch kernel | DeepSeek kernel | profile device time |
|---:|---:|---:|---:|---:|---:|---|
| 88 | 6.91 ms | 0.819 ms | 8.437x | 8 | 1 | 6.860 / 0.773 ms |
| 45 | 164 ms | 156 ms | 1.051x | 16 | 1 | 163.795 / 155.372 ms |
| 47 | 16.0 ms | 17.6 ms | 0.909x | 1 | 1 | 15.879 / 17.970 ms |
| 6 | 83.2 ms | 115 ms | 0.723x | 1 | 1 | 83.213 / 111.567 ms |
| 1 | 8.71 ms | 462 ms | 0.0189x | 1 | 1 | 8.533 / 461.821 ms |
| 55 | 35.9 ms | 6220 ms | 0.00577x | 1 | 1 | 30.282 / 6216.167 ms |
| 97 | 84.4 ms | 38200 ms | 0.00221x | 5 | 1 | 78.709 / 38348.590 ms |

## 6. 逐题性能归因

### Problem 88：融合消除中间张量——已由消融实验证实

原始 PyTorch 代码把 `pow`、4 次乘法、2 次加法和 `tanh` 表达为独立算子。profiler 观察到 8 个设备 kernel，总 device time 6.860 ms。DeepSeek 在一个 `new_gelu_kernel` 内完成全部运算，只读一次输入、写一次输出，device time 0.773 ms。

对同一输入执行控制变量实验：

| 实现 | 20 次均值 | 数值关系 |
|---|---:|---|
| 原始逐算子表达式 | 6.960 ms | reference |
| `F.gelu(approximate="tanh")` | 0.822 ms | allclose，最大绝对差 `1.19e-7` |
| DeepSeek kernel（正式值） | 0.819 ms | 通过 5/5 正确性 |

结论：8.44x 加速来自**表达式融合与中间张量消除**。DeepSeek 并没有击败 PyTorch 的 fused GELU；它只是把未融合的 reference 改写成了与 PyTorch fused GELU 性能相当的单 kernel。

### Problem 45：相同数据遍历下界——近似持平

输入包含约 42.95 亿个 fp32 元素，kernel size 和默认 stride 均为 11，因此窗口基本不重叠。两种实现都必须读取接近全部输入，并为每个输出做 121 项累加，几乎没有可利用的跨窗口复用。

PyTorch 的 MUDNN `AvgPool2dKernel` 在该超大输入上内部记录为 16 次调用；DeepSeek 使用一次扁平 grid launch。二者 device time 为 163.795 与 155.372 ms，仅相差约 5%。这里一次 launch 的形状特化减少了一些调度开销，但主体仍由相同的数据读取与累加决定。

证据边界：没有 `mcu` 内存带宽计数器，因此“接近带宽/遍历下界”是由输入规模、无重叠窗口和时间比例共同支持的强推断，不是硬件计数器直接测量。

### Problem 47：相同单 kernel，差异在索引常数项

两边均只启动一个 reduction kernel，并读取约 21.47 亿个 fp32 元素。DeepSeek 为每个输出线程串行处理 4096 项；相邻输出线程在每个归约步读取相邻地址，因此仍有合并访存，但循环内每次重新执行 64 位乘加来构造 `input_index`。

原 DeepSeek 为 17.6 ms，PyTorch MUDNN reduction 为 16.0 ms。已有控制变量版本仅改为首地址加指针步进并四次展开，保持累加顺序，正式时间降至 14.4 ms。由此可将原实现至少一部分慢化明确归因于**重复的地址计算和循环控制**，而不是算法复杂度或 kernel launch。

### Problem 6：large-K GEMM 的形状让朴素实现只中度落后

矩阵形状为 `256 x 524288` 乘 `524288 x 256`，约 68.72 GFLOP。PyTorch 调用单个 `musa_asm_sgemm_nn_512_128x128_8x4_epilogue`，DeepSeek 也只启动一个 kernel，但每个线程计算一个输出，K 循环八次展开且没有显式 shared-memory tile。

profile 时间为 83.213 与 111.567 ms。DeepSeek 仍然落后，但没有像方阵 GEMM 那样恶化几十倍。源码显示同一线程块内：A 的同一行元素会被多个列线程读取，B 的相邻列由相邻线程读取；cache/broadcast/coalescing 可能缓解部分重复访存。同时 M、N 仅为 256，也限制了原生 SGEMM 的并行规模。

“cache/broadcast 缓解”属于源码支持的推断；需要 Moore Perf Compute 的 cache hit、memory throughput 和 occupancy 指标才能最终确认。

### Problem 1：无分块 GEMM 丢失数据复用

`4096 x 4096` 方阵乘约 137.44 GFLOP。PyTorch 使用单个汇编 SGEMM，profile 为 8.533 ms，约 16.1 TFLOP/s；DeepSeek 每线程独立计算一个输出，串行遍历 K=4096，没有 shared-memory tiling 或寄存器级输出块，profile 为 461.821 ms，约 0.298 TFLOP/s。

每个 A/B 元素本可服务一整个输出 tile；DeepSeek 却从“输出元素”视角反复读取矩阵行和列，失去了 GEMM 最关键的数据复用。两边 launch 都为 1，故 54x 差异不能归因于 Python 或 launch，而是 GEMM kernel 算法映射的差异。

### Problem 55：直接卷积没有输入和权重 tile 复用

该卷积约 614.86 GFLOP。PyTorch 使用单个 `musa_asm_nchw_conv_256_128x128_epilogue_precompute`，profile 30.282 ms，约 20.3 TFLOP/s；DeepSeek 也是单次 launch，但一个线程串行计算一个输出，遍历 64 个输入通道和 `3 x 3` kernel，profile 6216.167 ms，约 0.099 TFLOP/s。

相邻输出共享大部分输入窗口，同一权重也会用于大量 batch/空间位置。DeepSeek 没有把输入或权重分块放入 shared memory/register，也在内层保留 64 位索引和边界判断。205x profile 差异发生在单 kernel 内部，主障碍是**数据复用和专用卷积微内核缺失**。

### Problem 97：融合不能弥补低效矩阵乘

该 attention 的两个矩阵乘合计约 1.10 TFLOP。PyTorch 当前不是 FlashAttention 路径，而是数学分解：两组汇编 SGEMM（profiler 各记录 2 次内部调用）加一个 softmax，共 5 个设备 kernel，合计 78.709 ms。

DeepSeek 将整个 attention 合并到一个 kernel，但每个 block 只负责一条 query row：

- 为每个 query row 重新遍历整块 K；
- 为每个输出列重新遍历整块 V；
- 点积和输出累加采用标量串行循环；
- 多次使用 block-wide shared-memory reduction 和同步；
- 不在多个 query row 之间复用 K/V tile。

结果为 38348.590 ms，约比 profile 中的 PyTorch 慢 487x。结论是：减少 launch 和融合 softmax 只节省了中间写回；损失 GEMM 的分块复用和高吞吐微内核后，代价远大于融合收益。

## 7. 无法计时和正确性障碍

### Problems 63、76、87

三题均通过 5/5 正确性，但 `ModelNew.forward()` 在返回前执行：

```python
x.set_(x.new_empty(0))
```

这会原地销毁调用者持有的输入。正确性阶段每个 trial 重建输入，所以能够通过；性能阶段复用同一输入，第二次调用即报“input must be 3D/4D”。应先移除输入 mutation，并通过分块 correctness check 或调整评测器的内存生命周期解决显存问题，再谈性能。

### Problem 72

baseline 中的 MUSA grouped ConvTranspose3d reference 输出异常，候选相对 CPU 被记录为 bit-exact，但相对当前 torch_musa/MUDNN reference 为 0/5。该题首先是后端正确性与 oracle 选择问题，不应计算 speedup。

## 8. 后续使用官方工具时要验证的指标

升级到匹配的 Moore Perf 环境后，优先对 Problems 1、55、97 采集：

- LaunchStats：threads/block、registers/thread、shared memory、理论/实测 occupancy；
- MemoryWorkloadAnalysis：DRAM/L2 吞吐、cache hit、load efficiency、shared bank conflict；
- SpeedOfLight：计算管线与内存管线利用率；
- Roofline：实测 FLOPs、bytes 和 arithmetic intensity；
- stall 原因：memory dependency、execution dependency、barrier。

预期验证：Problem 1/55/97 的 DeepSeek kernel 应表现为低计算吞吐、低数据复用和显著的长串行依赖；Problem 88 的 fused kernel 应显著减少 DRAM traffic；Problem 6 的 cache/broadcast 是否确实缓解重复读取则仍需这些指标确认。

## 9. 复现

```bash
MAX_JOBS=16 PYTHONWARNINGS=ignore \
PYTHONPATH=/root/autodl-tmp/KernelBench/src \
/root/miniconda3/envs/kernelbench-musa/bin/python \
analysis/pytorch_vs_deepseek_musa/profile_operator_pair.py 88
```

将末尾题号替换为 `45 47 6 1 55 97`。结果写入 `profiles/`：

- `problem_*_profile_summary.json`：去除 ATen 父事件重复后的结构化摘要；
- `problem_*_pytorch_trace.json`：PyTorch Chrome trace；
- `problem_*_deepseek_trace.json`：DeepSeek Chrome trace。

Chrome trace 可在 Perfetto 或 Chromium trace viewer 中打开。正式速度仍以 baseline 的 100 次 MUSA event 计时为准，profiler 单次时间只用于结构归因。
