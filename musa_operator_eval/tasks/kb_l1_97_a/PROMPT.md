# Scaled dot product attention（A 类：算子实现）

## 目标

Q, K and V arrive already split and already headed; the whole computation is the attention itself.
自己写 Device kernel 完成核心计算。禁止调用任何库算子。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与 reference `Model` 一致。兜底 kernel 可用 `kernelbench.musa_extension.load_inline` JIT 编译 `.mu` 源码。

## 必须做

- 把 QK^T、缩放、Softmax、P·V 全部放在 Device kernel 内完成。
- 支持动态 B / H / S / D，包括 S 与 D 非 tile 对齐。

## 禁止做

- 调用 muDNN / muBLAS / SDPA / ATen 完成任何一段核心计算。
- 把数据拷回 Host 计算。
- 按 shape 查表或返回固定输出。

## 工程提示

- head_dim 在本组内可以很大，q/o 的行状态会挤压可用的 band 宽度。
- 在线 Softmax 避免物化完整 score 矩阵。
- 非对齐的 S / D 是隐藏 Case 的常客。

## 评分口径

```text
speedup = upstream_musa_baseline_latency / submission_latency
```

A 类结果单独排名，不与 B 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。
