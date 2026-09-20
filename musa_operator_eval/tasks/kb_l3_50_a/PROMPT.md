# ReLU causal self-attention（A 类：算子实现）

## 目标

用 ReLU 替代 Softmax 的因果自注意力：遮挡后的分数直接 ReLU，没有任何分母，输出是加权和而不是加权平均。
自己写 Device kernel 完成核心计算。禁止调用任何库算子。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与 reference `Model` 一致。兜底 kernel 可用 `kernelbench.musa_extension.load_inline` JIT 编译 `.mu` 源码。

## 必须做

- 把 QKV 投影、QK^T+ReLU 遮挡、P·V 三段都放进 Device kernel。
- ReLU 必须作用在遮挡之后的分数上，顺序不能与遮挡互换。
- 注意 reference 不应用输出投影，提交也不得自行加上。

## 禁止做

- 调用 muDNN / muBLAS / SDPA / ATen 完成任何一段核心计算。
- 把数据拷回 Host 计算。
- 自作主张补一个 Softmax 归一化或输出投影。

## 工程提示

- 没有分母意味着不需要在线 rescale，整条流水线是纯流式的。
- ReLU 会把大量分数压成 0，跳过全零的 K/V tile 是这里唯一有价值的稀疏性。
- 因果遮挡在 band 内是一条斜边，ReLU 之后再处理就晚了，顺序要对。

## 评分口径

```text
speedup = upstream_musa_baseline_latency / submission_latency
```

A 类结果单独排名，不与 B 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。
