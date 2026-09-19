# 阶段 05：评测器控制面

状态：已完成并验证  
日期：2026-09-19

## 已交付

- 严格短路的评测入口 `evaluator/evaluate.py`。
- Schema、冻结文件、编辑区域和意外文件检查。
- B 类源码禁用项检查：ATen/PyTorch、CUDA 残留、动态加载、联网调用、版本字符串分派。
- host-only build 阶段。
- 原始 FP16/BF16/FP32 张量解码、SHA-256/尺寸校验和容差比较。
- 可选 MUSA runner 的公开正确性执行入口。
- B 类分派轨迹与 `expected_path` 逐 Case 核对。
- `pass/fail/incomplete` 报告；未运行硬件阶段只能是 `incomplete`。

## 当前退出码

- `0`：所有必需阶段通过。
- `1`：至少一个已执行阶段失败；报告内保存具体阶段码。
- `3`：控制面通过，但缺少 MUSA runner、隐藏资产或性能适配器。
- 阶段码：Schema 10、静态审计 20、构建 30、公开正确性 40、分派轨迹 50。

## 本地验证命令

```bash
python musa_operator_eval/evaluator/evaluate.py \
  --control-plane-only \
  --report runs/musa_eval_control/report.json
python musa_operator_eval/tools/validate.py \
  runs/musa_eval_control/report.json --kind report
python -m unittest discover -s musa_operator_eval/tests -v
```

验证结果：控制面报告通过 Schema；未提供 MUSA runner 时状态为 `incomplete`、退出码为 3；前五阶段累计 16 项测试全部通过。

## 尚未完成

- private manifest 挂载后的隐藏/边界/泛化执行。
- 构建产物实际链接库和导出符号检查。
- MUSA event 性能测量及六类 baseline 汇总。
- S4000/S5000 上的 native library adapter、expert dispatcher 和 fallback kernel。

## 下一阶段

补齐 private Case 模板（仅在被 Git 忽略的本地目录生成）、baseline 采集格式、候选题 1.3 门禁和硬件执行清单；随后进入必须在 MUSA 主机完成的 native adapter 实现与测量阶段。
