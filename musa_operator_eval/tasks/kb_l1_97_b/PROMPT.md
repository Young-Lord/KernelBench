# Scaled dot product attention（B 类：库集成）

## 目标

Q, K and V arrive already split and already headed; the whole computation is the attention itself.

优先使用任务白名单内的融合库能力；融合路径不接受当前配置时，退到库组合，最后才自己写 Device kernel 兜底。
三条路径必须符合同一份语义。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与 reference `Model` 一致，并且**每个 Case 写出一条分派轨迹**。

## 分派轨迹

评测端会设置两个环境变量：`KB_DISPATCH_TRACE`（往这个文件追加 JSON Lines）与 `KB_DISPATCH_CASE_ID`（当前 Case id，必须填进 `case_id` 字段）。
三个字段 `case_id` / `selected_path` / `probe_status` 都必须有且非空；`selected_path` 只能取 `fused_library` / `library_composition` / `custom_fallback`。同一个 Case 的多次 forward 必须报同一条路径。

## 必须做

- 按 fused_library → library_composition → custom_fallback 的优先级选择路径。
- 通过运行时能力返回值判断配置是否被接受，**不得按版本号硬编码**。
- 库拒绝配置或库语义不等价时，走正确的兜底路径。
- 保持 reference 的语义（见 `semantics.json`）。
- 处理非对齐的动态形状，不能只支持公开 Case。

## 禁止做

- 调用白名单之外的库、ATen 或 PyTorch 计算接口完成核心计算。
- 根据驱动、Toolkit 或库版本字符串硬编码分派。
- 把输入复制回 Host 执行张量数学。
- 联网、动态下载或动态加载外部实现。
- 全局降级成朴素实现：那样分派轨迹对得上，性能分拿不到。

## 工程提示

- 融合库路径通常具有 dtype、布局、head dimension、属性组合和设备相关约束；以运行时探测结果为准。
- 库组合路径覆盖面通常更广，但可能产生中间张量和额外 launch。
- 未知配置必须探测，不能默认融合库支持。
- `agent_reference/sdpa_forward_capabilities.json` 提供结构化的通用能力说明，不代表任何具体配置必然受支持。
- 目标环境记录在 `environments/musa-5f9d7b9dd1233a68.public.json`。

## 评分口径

```text
speedup = expert_dispatch_latency / submission_latency
```

先过门禁：全部隐藏正确性通过，且每条分派轨迹与评测端预期路径一致。B 类结果单独排名，不与 A 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。
