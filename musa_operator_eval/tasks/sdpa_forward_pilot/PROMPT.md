# SDPA Forward Library Dispatch

## 目标

实现一个面向动态输入的 SDPA forward runner。优先使用任务白名单内的融合库能力；融合路径不接受当前配置时，选择允许的库组合或自研 Device fallback。所有路径必须符合 `semantics.json` 的同一语义。

## 交付物

- 完成 `starter/src/dispatch.cpp` 中标记的分派区域。
- 完成 `starter/src/launch.mu` 中标记的 Host 启动区域。
- 完成 `starter/src/fallback_kernel.mu` 中标记的兜底 kernel 区域。
- runner 为每个 Case 输出 `output`，可选输出 `lse`。
- runner 为每个 Case 写出一条分派轨迹。

## 必须做

- 按 `fused_library → library_composition → custom_fallback` 的优先级选择路径。
- 通过运行时能力返回值判断配置是否被接受。
- 在库拒绝配置或库语义不等价时使用正确的兜底路径。
- 保持 top-left causal、窗口方向、FP32 Softmax 累加和全 mask 行语义。
- 处理非 tile 对齐的动态形状，不能只支持公开 Case。
- 只修改 `starter_manifest.json` 声明的编辑区域。

## 禁止做

- 调用白名单之外的库、ATen 或 PyTorch 计算接口。
- 根据驱动、Toolkit 或库版本字符串硬编码分派。
- 将输入复制回 Host 执行张量数学。
- 联网、动态下载或动态加载外部实现。
- 将第三方 kernel 源码改名后内联到提交中。
- 用固定输出、Case ID 分支或 shape 查表绕过计算。
- 修改冻结的 ABI、主程序、张量 IO 或构建入口。

## 工程提示

- 融合库路径通常具有 dtype、布局、head dimension、属性组合和设备相关约束；以运行时探测结果为准。
- 库组合路径覆盖面通常更广，但可能产生中间张量和额外 launch。
- fallback 中可考虑分块 QK、在线 Softmax、K/V tile 复用和非对齐边界处理。
- 未知配置必须探测，不能默认融合库支持。
- `agent_reference/attention_capabilities.json` 提供结构化的通用能力说明，不代表任何具体配置必然受支持。

## 评分口径

首先要求全部隐藏正确性通过，且每条分派轨迹与评测端预期路径一致。通过门禁后，性能得分使用：

```text
speedup = expert_dispatch_latency / submission_latency
```

A 类结果和 B 类结果独立排名，不合并计分。

## 尝试预算

最多 20 次提交尝试；单次评测最长 7200 秒。任一审计、构建或正确性阶段失败都会短路后续性能阶段。
