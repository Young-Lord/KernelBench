# 阶段 03：语义契约、Reference 与公开 Case

状态：已完成并验证  
日期：2026-09-19

## 已交付

- 首个候选任务 `sdpa_forward_b_v0` 的 B 类任务契约。
- 冻结的 BHSD Attention 语义：top-left causal、窗口方向、FP32 累加、全 mask 行输出规则。
- 独立 NumPy CPU FP64 分块 reference，支持 MHA/GQA、交叉注意力、causal、窗口和显式 mask。
- 5 个不泄露库缺口的公开 smoke Case。
- 确定性输入生成器，使用 `PCG64(case.seed)`。
- FP16、FP32 和 IEEE BF16 原始二进制编码；每个张量记录字节数和 SHA-256。
- 冻结 ABI 文档与公开 golden 生成器。

## 约束

任务暂按 B 类候选落盘，但只有在硬件能力探测证明“大部分配置可由融合库覆盖”、且朴素组合与 expert 分派延迟比不低于 1.3 后才能转为正式题。未通过门禁时应重设 Case 或撤销 B 类任务，而不是修改实测结论。

## 验证命令

```bash
python musa_operator_eval/tools/validate.py \
  musa_operator_eval/tasks/sdpa_forward_pilot/task.json --kind task
python musa_operator_eval/tools/validate.py \
  musa_operator_eval/tasks/sdpa_forward_pilot/semantics.json --kind semantics
python musa_operator_eval/tools/validate.py \
  musa_operator_eval/tasks/sdpa_forward_pilot/public_cases.json --kind cases
python musa_operator_eval/tools/generate_cases.py
python -m unittest discover -s musa_operator_eval/tests -v
```

验证结果：3 份正式文档通过 Schema/业务规则校验，5 个公开 Case 已生成，相关测试及前序测试共 10 项全部通过。

## 下一阶段

建立 A/B 共用 native Starter 骨架：冻结 manifest IO、公共头文件和主程序；B 类提供可编辑 dispatcher、分派轨迹接口和 fallback kernel 标记区。无 MUSA SDK 的本地环境先验证 host-only manifest IO，MUSA 编译在目标主机完成。
