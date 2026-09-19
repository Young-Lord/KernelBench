# MUSA 算子评测集实施进展

更新日期：2026-09-19

| 阶段 | 状态 | 总结 |
| --- | --- | --- |
| 01 控制面与 Schema | 已完成并验证 | `phase_01_schema_and_scaffold.md` |
| 02 环境与能力矩阵 | 工具已完成；硬件实测待执行 | `phase_02_environment_and_capability.md` |
| 03 语义、Reference、公开 Case | 已完成并验证 | `phase_03_semantics_reference_and_cases.md` |
| 04 Native Starter | host-only 已完成并验证；MUSA 构建待执行 | `phase_04_native_starter.md` |
| 05 Evaluator 控制面 | 已完成并验证 | `phase_05_evaluator_control_plane.md` |
| 06 发布门禁与交接 | 本地已完成并验证；正式 admission 待硬件 | `phase_06_release_gates_and_handoff.md` |
| 07 Agent 参考与题面 | 已完成并验证 | `phase_07_agent_reference_and_prompt.md` |

## 当前结论

首个 `sdpa_forward_b_v0` 候选题已经具备版本化契约、公开数据、独立 reference、冻结 ABI、Starter、审计器、评测控制流和发布隔离。它仍是候选题而非正式发布题，因为 B 类资格必须由目标 MUSA 机器上的库覆盖、缺口证据、expert 分派及 `naive/expert >= 1.3` 实测共同确认。

## 本地质量状态

- 22 项单元测试通过。
- 10 个正反向 Schema fixture 判定符合预期。
- 5 个公开 smoke Case 及 golden 已确定性生成。
- C++17 host-only Starter 编译和裸二进制往返通过。
- 控制面评测报告合法，缺少硬件时明确为 `incomplete`。
- `private/` 默认不进入 Git；公共包导出器排除来源、能力证据、隐藏数据和基线。

## 下一入口

在 MUSA 主机从仓库根目录打开 `musa_operator_eval/HARDWARE_RUNBOOK.md`，按顺序完成环境快照、三条库路径能力探测、native adapter、隐藏 Case、六类基线和 admission。不要先填 baseline 或手工指定库缺口。
