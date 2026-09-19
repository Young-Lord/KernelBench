# 阶段 07：Agent Attention 参考与正式题面

状态：已完成并验证  
日期：2026-09-19

## 本阶段目标

补齐原指南 4.5 的 Agent 可读题面，并提供一份可机器读取但不泄露维护者来源、隐藏 Case 或精确库缺口的 Attention 能力参考。

## 已交付

- `schemas/agent_reference.schema.json`：Agent 参考 JSON 的结构和固定枚举。
- `agent_reference/attention_capabilities.json`：三层能力类别、运行时分派规则、语义摘要和通用优化提示。
- `tasks/sdpa_forward_pilot/PROMPT.md`：目标、交付物、必须做、禁止做、工程提示、评分和尝试预算。
- `tools/audit_agent_package.py`：检查仓库 URL、完整 commit、维护者来源路径、具体来源项目名和隐藏资产路径。
- 公共打包器现在强制包含题面和 Agent reference，并在完成复制后自动执行泄漏审计。
- 新增 Agent reference Schema、公共包内容和故意泄漏负例测试。

## 信息边界

Agent reference 可以公开：

- 融合库、库组合、自研 fallback 三类能力；
- dtype/layout/head dimension/属性组合可能受限这一类通用事实；
- 必须依据运行时返回值探测；
- 统一语义和通用 kernel 工程建议。

Agent reference 不公开：

- 上游仓库、commit、URL、版本和具体函数名；
- 哪一个精确 shape 会触发库缺口；
- 隐藏 Case、expert 源码或 expert 的精确分派表。

## 验证命令

```bash
python musa_operator_eval/tools/validate.py \
  musa_operator_eval/agent_reference/attention_capabilities.json \
  --kind agent_reference
python -m unittest discover -s musa_operator_eval/tests -v
```

## 验证结果

- Agent reference 通过专用 Schema 校验。
- 全部 22 项单元测试通过。
- 37 个 JSON 文件均可解析。
- 已实际导出包含 53 个文件的 Agent 公共包。
- 导出包通过独立泄漏审计，未包含来源仓库、完整 commit、URL、维护者源码路径或隐藏资产。
- 导出包中的 `agent_reference/attention_capabilities.json` 再次通过 Schema 校验。

## 下一阶段

在 MUSA 主机执行 capability plan，并把实测结果只写入维护者侧证据文件。Agent reference 继续保持通用描述；不得用实测到的精确缺口反向扩充公开参考。
