# Vision attention block（A 类：算子实现）

## 目标

一个视觉注意力块：把 (B,C,H,W) 拉平成序列，做带投影的自注意力，再经 LayerNorm(输出+残差) 还原回图像布局。
自己写 Device kernel 完成核心计算。禁止调用任何库算子。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与 reference `Model` 一致。兜底 kernel 可用 `kernelbench.musa_extension.load_inline` JIT 编译 `.mu` 源码。

## 必须做

- 把 QKV 投影、QK^T+Softmax、P·V、输出投影都放进 Device kernel。
- LayerNorm 的归约与残差相加的顺序必须与 reference 一致。
- 支持非 2 的幂、非 tile 对齐的 H*W 序列长度。

## 禁止做

- 调用 muDNN / muBLAS / SDPA / ATen 完成任何一段核心计算。
- 把数据拷回 Host 计算。
- 假设 H*W 是某个 tile 的整数倍。

## 工程提示

- 这里没有 mask，整块 score 矩阵都是有效的，band 覆盖全部 K/V。
- 序列长度由 H*W 决定，H 或 W 非 2 的幂时它会很不对齐。
- 把 (H*W, B, C) 的布局转换成寄存器友好的块状访问是主要工作量。

## 评分口径

```text
speedup = upstream_musa_baseline_latency / submission_latency
```

A 类结果单独排名，不与 B 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。
