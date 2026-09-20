# MUSA 算子评测集 v0 周工作汇报

> **修订说明（2026-09-20）**
> 本文描述的是第一轮实现。其中的原生 C++ ABI、7 类 JSON Schema、fixtures、并行评测器
> 和打包/泄漏审计脚本已经删除：它们重复实现了 KernelBench 框架已有的能力
> （`eval.py`、`timing.py`、`kernel_static_checker.py`、`musa_extension.py`），
> 而冻结的 ABI 又与 B 类"必须调用 muDNN"的前提相冲突。
> 保留下来的只有 KernelBench 没有的部分：能力探测、环境采集、隐藏 Case 生成和候选门禁。
> 下文凡提到已删除组件的段落仅作为决策记录保留，当前布局见 `musa_operator_eval/README.md`。

## 一、本周工作主题

**将 MUSA 算子评测集设计文档转换为可执行规范，并完成首个 SDPA 候选题的本地评测闭环。**

原需求文档定义了 A 类 kernel 实现和 B 类库集成两条评测轨，但主要是人工规范。本周工作的重点不是直接给出某个 kernel 的最终性能结果，而是先完成评测集的工程基础，使任务能够被版本化描述、自动校验、确定性生成、静态审计和安全发布。

首个纵向样例选择 `sdpa_forward_b_v0`，当前定位为 B 类候选题。最终是否满足 B 类条件，仍需要真实 MUSA 环境中的库覆盖、缺口和性能证据确认。

## 二、本周主要成果

| 成果 | 工作内容 | 对应证据 |
| --- | --- | --- |
| 需求工程化 | 将 A/B 两级评测要求转换成版本化、机器可校验的规范 | 7 类 JSON Schema、正反向 fixture、校验工具 |
| 首题纵向切片 | 完成 `sdpa_forward_b_v0` 候选题的公开部分 | task、semantics、PROMPT、公开 Case |
| 独立正确性基准 | 实现 CPU FP64 分块 SDPA reference | causal、window、GQA、全 mask 行测试 |
| 冻结 ABI 与 Starter | 建立不依赖 libtorch/ATen 的原生接口 | C++17 IO、SHA-256、编辑区域、冻结锁 |
| 评测控制面 | 实现 Schema、审计、构建、数值比较、轨迹检查和报告流程 | evaluator、compare、短路状态 |
| 发布隔离 | 区分维护者资产和 Agent 可见资产 | 脱敏能力 JSON、公共包导出、泄漏审计 |

## 三、量化结果

本周完成的可量化产出包括：

- 7 类版本化 JSON Schema；
- 1 个 SDPA B 类候选任务；
- 1 份完整的 Agent 可读题面；
- 1 份脱敏 Attention 能力参考 JSON；
- 5 个公开 smoke Case 及对应输入和 golden；
- 1 个 CPU FP64 分块 reference；
- 1 套 FP16/BF16/FP32 裸二进制 ABI；
- 1 套 C++17 Native Starter；
- 1 套评测控制流和静态审计器；
- 1 套私有 Case 证据生成机制；
- 1 套六基线格式及 1.3 倍候选门禁；
- 1 套 Agent 公共包导出和泄漏检测；
- 22 项自动化测试全部通过；
- 37 个 JSON 文件全部可解析；
- 实际导出并审计了一个包含 53 个文件的 Agent 公共包；
- 7 份阶段性工作总结。

## 四、需求文档到机器契约

本周建立了以下 Schema：

- `task.schema.json`：任务契约；
- `semantics.schema.json`：Attention 语义契约；
- `cases.schema.json`：公开和隐藏 Case 清单；
- `environment.schema.json`：环境快照；
- `baseline.schema.json`：基线结果；
- `report.schema.json`：评测报告；
- `agent_reference.schema.json`：Agent 脱敏能力参考。

其中已经编码的关键规则包括：

- A 类不能出现库策略字段；
- B 类必须声明链接白名单、允许符号前缀、分派文件和分派轨迹；
- B 类公开 Case 不得暴露库缺口；
- B 类隐藏 Case 必须至少包含 3 个真实缺口，并覆盖至少 2 类缺口原因；
- B 类 speedup 分母必须是 expert dispatch；
- 所有契约均带版本号；
- 非法字段和跨层级字段会被校验器拒绝。

### 验证证据

```powershell
python musa_operator_eval/tools/validate.py --fixtures
```

预期所有合法和非法 fixture 均显示 `PASS`。非法 fixture 的 `PASS` 表示它被校验器成功拒绝。

## 五、首个 SDPA 候选任务

当前已经建立：

- `musa_operator_eval/tasks/sdpa_forward_pilot/task.json`；
- `musa_operator_eval/tasks/sdpa_forward_pilot/semantics.json`；
- `musa_operator_eval/tasks/sdpa_forward_pilot/PROMPT.md`；
- `musa_operator_eval/tasks/sdpa_forward_pilot/public_cases.json`。

任务中已经明确：

- 输入输出采用 BHSD；
- causal 对齐固定为 top-left；
- `window_left/window_right` 的方向和 `-1` 语义；
- Softmax 使用 max-shift；
- 累加采用 FP32；
- 全 mask 行输出为 0，LSE 为负无穷；
- dtype、动态轴、容差、可编辑文件和禁止项；
- B 类必须按融合库、库组合、自研 fallback 的顺序进行运行时分派。

当前任务仍是 B 类候选。只有在真实硬件上证明库覆盖、真实缺口及性能区分度后，才能正式 admission。

## 六、Reference、Case 和 Golden

本周实现了独立 NumPy CPU FP64 分块 reference，支持：

- MHA；
- GQA；
- `S_q != S_kv`；
- top-left causal；
- 左右窗口；
- 显式 mask；
- 全 mask 行；
- 稳定 Softmax；
- 分块计算，避免一次性构造过大的完整中间结果。

公开 Case 使用 `PCG64(case.seed)` 确定性生成。输入张量在量化为 FP16、BF16 或 FP32 后再计算 golden，从而保证 reference 使用的数值与实际写入二进制文件的数值一致。

每个张量 manifest 记录：

- 名称；
- 文件名；
- dtype；
- shape；
- layout；
- byte order；
- 字节数；
- SHA-256。

## 七、冻结 ABI 与 Native Starter

为了避免被测程序间接带入 ATen/libtorch，本周没有直接沿用 Python/PyTorch 模型接口，而是实现了：

```text
input/tensors.json + *.bin
            ↓
       native runner
            ↓
output/tensors.json + *.bin
```

Starter 中：

- 张量 IO、主程序、公共头文件和构建入口冻结；
- Agent 只能修改 dispatcher、Host launch 和 fallback kernel 的标记区域；
- 输入读取时验证 `nbytes` 和 SHA-256；
- 输出写入时重新计算 SHA-256；
- B 类每个 Case 输出一条 JSON Lines 分派轨迹；
- 分派轨迹包含 `case_id`、`selected_path` 和 `probe_status`。

本地已经验证：

- C++17 host-only 构建；
- 输入 manifest 解析；
- 原始二进制读取和写回；
- SHA-256 校验；
- 输入输出往返；
- 冻结文件修改检测；
- Agent 编辑区域完整性。

### 验证证据

```powershell
python musa_operator_eval/tools/freeze_starter.py
```

预期输出：

```text
starter lock verified
```

只要冻结的 IO、主程序、公共头文件或构建入口发生修改，该命令就会失败。

## 八、评测控制面

当前评测器已经实现：

```text
Schema
  → 静态审计
  → host-only build
  → 公开正确性接口
  → 分派轨迹检查
  → 隐藏评测接口
  → 性能接口
  → 统一报告
```

已经支持的审计包括：

- 修改冻结文件；
- 在标记区域之外修改代码；
- 增加未声明文件；
- 调用 ATen/PyTorch；
- CUDA 残留；
- 动态加载；
- 网络访问；
- 按版本字符串硬编码分派。

评测器还能够：

- 解码 FP16/BF16/FP32；
- 校验张量文件大小和 SHA-256；
- 比较 shape 和 dtype；
- 使用 atol/rtol 比较数值；
- 正确处理正负无穷和 NaN；
- 将分派轨迹与每个 Case 的期望路径匹配；
- 在阶段失败时短路后续执行；
- 区分 `pass`、`fail` 和 `incomplete`。

## 九、Agent 参考与发布隔离

本周补充了专供 Agent 使用的脱敏 Attention 能力参考：

```text
musa_operator_eval/agent_reference/attention_capabilities.json
```

该文件只公开：

- 融合库、库组合、自研 fallback 三类能力；
- dtype、layout、head dimension 和属性组合可能受限这一类通用事实；
- 必须依据运行时返回值进行能力探测；
- 统一 Attention 语义；
- 通用 kernel 优化提示。

该文件不会公开：

- 上游仓库名称；
- commit；
- URL；
- 版本；
- 具体库函数名；
- 精确库缺口配置；
- 隐藏 Case；
- expert 源码或精确分派表。

公共包导出器在完成复制后会自动执行泄漏审计，检查：

- 仓库 URL；
- 完整 commit hash；
- 维护者源码目录；
- 具体来源项目名；
- hidden/private 路径；
- Agent reference 中的禁止字段。

### 验证证据

```powershell
python musa_operator_eval/tools/audit_agent_package.py `
  runs/agent_package_phase07_20260919
```

预期输出：

```text
agent package disclosure audit passed
```

## 十、自动化测试

统一验证命令：

```powershell
python -m unittest discover -s musa_operator_eval/tests -v
```

当前预期结果：

```text
Ran 22 tests
OK
```

测试覆盖：

- Schema 正反例；
- A/B 字段约束；
- SDPA reference 数值；
- GQA；
- 全 mask 行；
- BF16 编解码；
- Case 确定性；
- Starter 编译和 IO；
- 冻结文件篡改；
- 禁用代码检查；
- 数值比较；
- 私有缺口数量和原因门禁；
- 1.3 候选 admission；
- Agent 公共包隔离；
- Agent reference Schema；
- 故意泄漏负例。

## 十一、控制面现场演示

运行：

```powershell
python musa_operator_eval/evaluator/evaluate.py `
  --control-plane-only `
  --report runs/group_meeting_demo/report.json
```

当前预期报告：

```text
schema              pass
static_audit        pass
host_build          pass
public_correctness  skipped
hidden_correctness  skipped
performance         skipped
overall             incomplete
```

`incomplete` 是刻意设计的状态，不代表控制面失败。它表示硬件正确性、隐藏评测或性能阶段尚未获得真实 MUSA 证据，防止未执行的硬件阶段被错误发布为通过。

随后可以验证报告格式：

```powershell
python musa_operator_eval/tools/validate.py `
  runs/group_meeting_demo/report.json --kind report
```

## 十二、当前完成边界

### 已完成并验证

- 需求文档到机器契约的转换；
- 首个 SDPA 候选任务的公开部分；
- Agent 可读题面和脱敏能力参考；
- CPU reference；
- 公开 Case 和 golden；
- 裸二进制 ABI；
- Native Starter 控制面；
- 本地静态审计；
- 评测编排和报告；
- 公共包导出和防泄漏机制。

### 尚未完成硬件验收

- B 类资格最终确认；
- S4000/S5000 正式环境快照；
- 真实库覆盖与真实库缺口；
- Native 融合库 adapter；
- expert dispatcher；
- fallback kernel；
- 正式隐藏评测；
- 六条正式基线；
- 1.3 倍性能 admission；
- MUSA 正确性、性能、显存、链接白名单和 sanitizer；
- S4000/S5000 完整发布验收。

这些内容必须依赖真实 MUSA 硬件和工具链。当前本地结果不会被描述为正式 benchmark 数据。

## 十三、下周计划

下周建议优先完成：

1. 在 S4000/`mp_22` 和 S5000/`mp_31` 上生成正式环境快照和能力矩阵；
2. 完成融合库、库组合、fallback 三条 native 路径及 expert dispatcher；
3. 基于实测缺口生成正式隐藏 Case；
4. 开始六条基线测量；
5. 执行 `naive/expert >= 1.3` 的 B 类 admission；
6. 将真实 runner 接入已有评测控制面。

需要组内确认的问题：

- S4000 和 S5000 使用一个任务还是拆分为两个任务；
- 哪些 MUSA 镜像作为正式评测环境；
- private 资产存放在独立仓库还是内部对象存储；
- 首个发布版本是否只覆盖 Attention；
- 正式基线运行所需的硬件时间和维护者权限。

## 十四、总结

本周完成了从需求文档到可执行评测规范的第一轮落地，形成了 SDPA 候选题、独立 reference、原生 ABI、Starter、评测控制面和 Agent 发布隔离，并通过 22 项自动化测试。下一阶段将在真实 MUSA 环境中补齐能力证据、native 实现、隐藏评测和正式基线。
