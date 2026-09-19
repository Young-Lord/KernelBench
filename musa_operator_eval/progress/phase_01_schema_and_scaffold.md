# 阶段 01：评测控制面与 Schema

状态：已完成并验证  
日期：2026-09-19

## 本阶段目标

把 `docs/musa_operator_evaluation_guide.md` 中依赖人工理解的字段约束，转换成可版本化、可执行检查的评测控制面。

## 已交付

- 建立 `musa_operator_eval/` 独立评测根目录。
- 建立任务、Attention 语义、Case、环境、基线、报告六类 JSON Schema，版本均为 `1.0.0`。
- A/B 分级约束：A 禁止 `library_policy`，B 强制要求库白名单、符号前缀、分派文件、轨迹字段和库缺口规则。
- B 类公开 Case 禁止暴露库缺口，且只能声明 `fused_library` 路径。
- B 类隐藏 Case 强制至少 3 个缺口、至少 2 类缺口原因。
- A/B 基线集合和 speedup 分母检查。
- 建立零第三方依赖的本地校验入口及 CPU 单元测试。
- 将 `musa_operator_eval/private/` 默认加入 `.gitignore`，只保留隔离说明。

## 设计决定

1. v0 语义 Schema 限定为 `sdpa_forward`，避免将 Attention 专属字段错误泛化到其他算子。
2. 目标架构同时允许 `mp_22` 和 `mp_31`；每个实际环境快照仍必须只记录一个确定架构。
3. 当前校验器覆盖项目使用到的 JSON Schema 子集，并单独执行跨文档业务规则；后续 CI 可再接标准 JSON Schema 实现做双重检查。
4. 隐藏资产不进入当前 Git 工作区的可跟踪集合。

## 验证命令

```bash
python musa_operator_eval/tools/validate.py --fixtures
python -m unittest discover -s musa_operator_eval/tests -v
```

验证结果：10 个正反向 fixture 判定符合预期，3 个单元测试全部通过。

## 下一阶段

实现环境采集器与 `sdpa_forward` 能力探测清单。无 MUSA 设备的机器应明确输出 `unavailable`，不得生成伪实测结果；在 S4000/S5000 环境运行同一工具后生成正式环境快照和能力矩阵。
