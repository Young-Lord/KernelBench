# MUSA Attention 内核优化实录（有效 / 无效改动对照）

> 平台：MUSA MTT S4000 / mp_22，fp32，KernelBench level1/level3 attention 题目。
> 全部结论基于 KernelBench eval（3/3 correctness，与 torch 参考同机对比）+ `perf_one` 全模型计时。
> 数据均为当时实测（该机有并行任务时会波动，见文末“测量噪声”）。

---

## 1. 背景与“题目”

| 题 | 结构 | 手写内核覆盖点 | 形状要点 |
|---|---|---|---|
| 43/44 | GPT-2 causal attention | fused QKV → softmax 注意力 | B128,H8,T512,d96，因果 |
| 50 | DeepSeek 式 relu 注意力 | 同上 relu 变体 | B16,H12,T1024,d64，因果 |
| 31 | VisionAttention（16k dense） | 多头注意核心 | N(seq)=16384,H4,d32，全连接 |
| 97 | level1 SDPA | 整个 scaled-dot-product | B32,H32,T512,**d1024** |
| 30 | Swin window attention | window+bias+mask 注意核心 | 窗 7×7=49,d32，多层多 head |
| 28/32 | ViT / Conv-ViT | 注意核心（退化，seq≈2） | 本来就快，无需改 |

正确性门槛：eval 3/3 PASS（3 个随机 trial 与 torch 参考在容差内一致）。

---

## 2. 各题修改历程与结论

### 43/44/50（band 基线 → flash-tile 分块）

**过程**
1. 提交基线是“band 版”：每 CTA 一个 query band + 整行 scores 进共享、块内固定 32 线程组归约。模型级 354/412/207ms。
2. 换“flash-tile”核心：每 CTA 一个 BM 行 band，KV 按 BN 块流式进共享，online-softmax rescale，每行 `ROW_LANES` 条 lane 各持若干输出列。
   - 初版默认 BM16/BN32/每行 8 lane = 128 线程 → 43 反而 668ms（比 band 慢），因 tile 小、归约/rescale 开销占比高。
   - 参数扫描（仅 kernel，43 causal 形状）核心规律：**每 CTA 线程数 128→256→512→1024 持续明显变好**；BN 也要适中（BN16@512 线程最优、BN32@1024 线程更优）。
3. 定稿 BM32/BN32/每行32lane（=1024 线程/CTA）：s43 kernel ~114ms（≈0.24×band 的 kernel-only 对比）。
   模型级最终 140.8 / 199.7 / 57.2ms（torch 47.7/106.2/19.3）。

**有效**
- “一个带多行共享 KV 流式 tile + 每行多 lane”整体思路。
- 线程数拉满到 1024（占满 CTA 上限）。
- 共享数组按 `d+1` / `BN+1` 加 padding 消除 bank 冲突。
- 因果只算上三角有效键（mask 语义 = masked_fill -inf / relu0）。

**无效 / 教训**
- BM=128 等更大 band 并不普适：共享预算只够小 head_dim（d≈32），d96 直接超 smem 启动失败（见 31 的 BM128 情况）。
- BN 过大（128）或 BM 超出 smem/线程上限 → 启动失败（表现为 0.0ms + 大 maxdiff，需数值校验才能察觉）。

---

### 31（VisionAttention，16k dense）

**过程**
1. 原实现：per (bh,row) dense kernel（1 CTA/行、256 线程、行内流式 KV）。模型 ~2945ms。
2. 移植 flash-tile dense（非因果）。初版 BM32/BN32/L32=670ms kernel；
   形状专用扫描（T=16384,d32）：**BM 越大越省 KV 往返**（每 band 载入一次 KV），BM128/BN32/每行8lane=523ms 最优（更大 BM 超 smem）。
   模型 ~527ms（0.144×torch）。
3. **再换架构 → 两段式 GEMM**（见 §3 有效模式）：QK^T（单级 tile，K 维仅 32）→ 全局行 softmax → P·V（行 tile × 32 列、按 key 分块）。
   模型 **291ms（0.26×torch）**，eval 3/3。

**有效**
- dense 长序列 + 小 head_dim 时，把 online rescale 换成“物化 S/P 的 GEMM + 全局 softmax”整体更快（靠高算术强度 + 库级 GEMM 式访存），尽管多出 S/P 各 ~8.6GB 中间张量。
- tile 大（128×128、每线程 8×8 微片）明显好于 64×64/4×4。

**无效 / 教训**
- BM128 配置在通用数值回归的 d96 用例会因 smem 不足失败——**配置必须按真实 head_dim 校验**，别拿通用回归小 shape 当准。
- 更大 BN（64/128）、BM256 在 d32 也都超 smem/线程上限。

---

### 97（SDPA，d=1024 —— 最难的形状）

**过程（多轮，逐步大改）**
| 版本 | 做法 | 结果 |
|---|---|---|
| 原始 | 1 CTA/(bh,row)，256 线程，每线程一 key 读 4KB 间隔 K 行 | 38176ms |
| v1 | 每行 8 warp 协同点积 + 合并访存 + 行内 register Q | 3712ms |
| v2 | 行组分带：8 行/CTA，warp=row，同一 K 行 L1 共享，V 每列一次供 8 行 | 1151ms |
| GEMM1 | 两段式：QK^T + softmax + PV，64×64 tile / 4×4 微片 | 320ms |
| GEMM2 | tile 128×128 / 8×8 微片 | 244ms |
| GEMM3 | + float4 向量化 staged 载入 | **191ms** |
| 试 R4/R16 / 2row-per-warp | 分带粗细调节 | 1.29~1.9s（更差） |
| 试 BK32 / 双缓冲 | GEMM 变体 | 419ms / 数值坏（弃） |

模型级 191ms（0.40×torch，原始 0.002× → ~200x 提升）。

**有效**
- 先把“非合并访存”修掉（v1 就 10x）：warp 沿 d 协同、lane 持 32 列、K 逐 key 全 warp 合并读。
- **行组复用 K/V**（v2，R=8）再压 ~3x：V 每列只读一次供整组行，K 靠同 key 时 8 个 warp 命中 L1。
- **GEMM 两段式才是质变**：把 1.1TFLOP 用高算术强度 SGEMM 消化（3.2→1.6→1.2→**0.9 TFLOP/s 有效**…实际相对提升 320→244→191ms）。
- 每线程 8×8 微片 & float4 载入两次 ~20% 递进收益（FMA/共享读比与指令数）。

**无效 / 教训**
- d=1024 无法用 flash-tile：q/o 状态就要 ~8KB/行，BM≤8 且每 CTA 线程过少 → 延迟饥饿 > 带宽，比原版还慢甚至触发看门狗。
- R16 / 每 warp 2 行：多行/warp 增寄存器、降占用，反效果。
- 更大 BK：分块迭代少但单块串行更久，更差。
- **双缓冲补丁出了数值 bug（竞争）且更慢**：加了双缓冲不等于更好，必须有数值校验把关；未验证前别采纳。

---

### 30（Swin window attention）

**过程**
1. 原实现：1 CTA/(window,head,row)，每行重复读整窗 K/V。整模型 110ms（attention kernel 实测约占一半，约 47%）。
2. **整窗内核**：1 CTA/(window,head)，Q/K/V 一次性进共享、整窗 N×N 分数共享数组、逐行 softmax、O=P·V 一次写回。
   attention 调用总时 52→**18.5ms**，整模型 **75.4ms（0.81×torch）**，eval 3/3。

**关键经验（反面教材）**
- 中途一次测量给出“整模型不变”并据此误判“内核非瓶颈” —— 事后查明是**文件被外部回滚、实际测的是旧内核**。教训：换内核后要先确认文件内容与编译哈希生效，再下结论（`rg` 内核标记 + 独立 hook 计时 kernel 占比）。
- 修正后：整窗内核把 attention 从 ~52→18.5ms（2.8x），模型 110→75.4ms。

**无效 / 教训**
- 整窗内核“串行逐行 softmax + 49 线程算 O”仍有提升空间（对 torch 仍差 4x 左右：torch 库 attention 约 ~4-8ms，我们 18.5ms），但相对收益已经到位；再往下要动 softmax 并行与 occupancy。

---

### 28 / 32
本就优于 torch（它们实际只对 seq≈2 的退化序列做 attention），未改。

---

## 3. 跨题“有效模式”总结（可复用的套路）

1. **先消除结构性浪费，再谈微调**：
   - 非合并访存（每线程一 key × 4KB 间隔）→ 沿连续维合并，通常就有 10x；
   - 同一数据被 N 个查询重复读 → 行/窗分带共享（K/V 复用倍数 ≈ 带宽度）。
2. **大而稠密的 attention 用“两段式 GEMM + 全局 softmax”**：
   - 物化 S/P 虽多写多读，但把算力从 ~0.3 TFLOP/s 抬到 ~2-6 TFLOP/s，整体净赚 1.8-3x；
   - 适用条件：序列/嵌入足够大值得 GEMM 化；中间 S/P 大小能装下显存。
3. **SGEMM 调节杠杆（按收益排序，实测）**：
   - 寄存器微片 4×4 → 8×8（~25%）；
   - float4 向量化共享 staged 载入（~20%）；
   - 线程数/tile 尺寸贴上限（256~1024 线程、tile 128×128）；
   - 共享行 padding 避免 bank 冲突。
4. **小窗口注意力（N≈49）**：整窗单 CTA 完胜逐行（CTA 数降 N 倍 + KV 只读一次）。
5. **数值校验是护栏**：每版 kernel 都要过（a）小 shape 对 torch 公式 maxdiff（<1e-3，通常 ~1e-7）与（b）eval 3/3；启动失败/竞争往往表现为 0.0ms + 大 maxdiff 或偶发大 diff。
6. **按真实 shape 选择/校验配置**：d 决定 smem 预算（BM×sd）；同一套“最优”配置不能盲目复制到不同 head_dim/seq 的题。

---

## 4. 无效 / 陷阱清单（每一条都付出过实测成本）

| 改动 | 现象 | 原因 |
|---|---|---|
| flash-tile 用于 d=1024 | 比原版还慢 / 看门狗超时 | tile 太小 + 1 CTA/SM 延迟饥饿 |
| BM 超出 smem 上限（d96 用 BM128 等） | 启动失败（0.0ms+大 diff） | smem 预算 = f(BM, d) |
| BN 过大（p31 BN64/128） | 同上 | 同上 |
| 每 CTA 线程 > 1024（BM×lane） | 启动失败 | 架构线程上限 |
| p97 GEMM BK16→32 | 244→419ms | 单块串行更长 |
| p97 双缓冲 | 数值坏 + 更慢 | 手写 barrier 竞争，未过数值关 |
| p31/p97 更大 BM、多行/warp | 更慢（1.29~1.9s） | 寄存器/占用下降 |
| “整窗内核无收益”误判 | 结论错误 | 文件被回滚，测的是旧内核（先验 file hash + 生效再下结论） |
| 通用数值回归里跑非本题 shape | 误报 FAIL | 配置对 d>预算失效，不代表本 shape 错（校验时用真实 shape + 一组小 shape 都要看） |

---

## 5. 测量与协作环境注意事项

- 同一批优化在该机有**并行任务时耗时波动明显**（同一 kernel 可 168↔292ms），比大小要用中位数 & 稳定复测；评估“是否更优”前先确认无并发。
- 工具 stdout 在大输出时会被截断；**把结果重定向到文件再读**。
- 最终“成绩”口径统一用 `perf_one.py`（同一进程加载 ref+ours，同种子初始化，3-5 iter 平均），并与 eval 3/3 绑定。
- 代码库存在外部进程对未提交 model 文件做同步/回滚的现象：重要改动尽早 commit，并在每次评估前核对文件内容。

---

## 6. 最终成绩（均 eval 3/3 PASS，fp32）

| 题 | 起点 | 终点 | vs torch |
|---|---|---|---|
| 43 | ~354ms | 140.8ms | 0.34x |
| 44 | ~412ms | 199.7ms | 0.53x |
| 50 | ~207ms | 57.2ms | 0.34x |
| 31 | ~2945ms | 291.4ms | 0.26x |
| 97 | ~38176ms | 191.0ms | 0.40x |
| 30 | ~110ms | 75.4ms | 0.81x |
| 28/32 | — | 不变 | >1x（原就快于 torch） |

相关提交：`898244a`（43/44/50/31 flash-tile）、`edee502`/`bbb741b`/`2084d16`（97 GEMM 系列）、`ba8d674`（31 两段式 GEMM）、`60d82eb`（30 整窗内核）。原始 band 基线在 `c6b6eb0`。
