# MinGPT causal attention block（A 类：算子实现）

## 目标

一个 MinGPT 因果注意力块：融合的 QKV 投影、分头、缩放 QK^T、top-left 因果遮挡、Softmax、P·V、输出投影。
自己写 Device kernel 完成核心计算。禁止调用任何库算子。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与 reference `Model` 一致。兜底 kernel 可用 `kernelbench.musa_extension.load_inline` JIT 编译 `.mu` 源码。

## 必须做

- 把 QKV 投影、QK^T+Softmax、P·V 三段都放进 Device kernel。
- 因果遮挡必须与 reference 的 tril 对齐方式一致（含对角）。
- 支持动态 B / S / C / H，D = C/H 由输入决定。

## 禁止做

- 调用 muDNN / muBLAS / SDPA / ATen 完成任何一段核心计算。
- 把数据拷回 Host 计算。
- 只处理 C/H 对齐或 S 为 tile 整数倍的形状。

## 工程提示

- 融合的 QKV 投影是一次 GEMM，注意写出 (B*S, 3C) 的布局。
- 因果遮挡在 band 内表现为对角线上的一条斜边，处理它比处理整块 mask 便宜得多。
- 在线 Softmax 避免物化 (B, H, S, S) 的 score 矩阵。

## 评分口径

```text
speedup = upstream_musa_baseline_latency / submission_latency
```

A 类结果单独排名，不与 B 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。
