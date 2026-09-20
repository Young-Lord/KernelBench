# MiniGPT transformer block（A 类：算子实现）

## 目标

完整的 MiniGPT Transformer block：因果注意力、两层 LayerNorm、4 倍扩张的 MLP（tanh 近似 GELU）以及两条残差路径。
自己写 Device kernel 完成核心计算。禁止调用任何库算子。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与 reference `Model` 一致。兜底 kernel 可用 `kernelbench.musa_extension.load_inline` JIT 编译 `.mu` 源码。

## 必须做

- 把注意力三段与 MLP 的两次投影都放进 Device kernel。
- LayerNorm 的均值方差统计必须与 reference 在同一精度下一致。
- 残差相加的位置必须与 reference 完全一致（norm 之后进子层，不是之前）。

## 禁止做

- 调用 muDNN / muBLAS / SDPA / ATen 完成任何一段核心计算。
- 把数据拷回 Host 计算。
- 把 MLP 或 LayerNorm 留给框架算子。

## 工程提示

- MLP 的 4 倍扩张让它的 FLOPs 与注意力同量级甚至更高，不要只优化注意力。
- LayerNorm 的归约与 Softmax 的归约可以复用同一套 warp reduction 代码。
- GELU 用的是 tanh 近似而不是 erf，两者数值不同，照抄 reference 的公式。

## 评分口径

```text
speedup = upstream_musa_baseline_latency / submission_latency
```

A 类结果单独排名，不与 B 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。
