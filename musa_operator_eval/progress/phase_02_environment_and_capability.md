# 阶段 02：环境采集与 SDPA 能力矩阵

状态：实现完成，硬件实测待在 MUSA 主机执行  
日期：2026-09-19

## 已交付

- `tools/collect_environment.py`：采集完整和脱敏环境快照；关键字段缺失时只输出 `incomplete` 探针，不伪造正式快照。
- `tasks/sdpa_forward_pilot/capability_plan.json`：12 个覆盖 dtype、布局、交叉注意力、GQA、非对齐 head dim、窗口语义的能力探针。
- `tools/probe_sdpa.py`：可在 MUSA 主机运行的 torch_musa SDPA 探测入口；无 MUSA 时明确返回 `unavailable`。
- CPU 单元测试检查探针 ID 唯一性、维度完整性和不可用状态的显式表达。

## 重要边界

1. torch_musa 高层调用成功不能证明走了融合库，因此探针将 `selected_route` 保持为 `unverified`。
2. muDNN 和 MATE 的直接能力探测必须由后续 native adapter 提供；当前不会把未测能力标成支持。
3. Windows 开发机产生的 `environment_probe.json` 只是工具自测，不是正式评测环境。

## MUSA 主机执行命令

```bash
python musa_operator_eval/tools/collect_environment.py \
  --output-dir runs/musa_env \
  --architecture mp_22 \
  --device-name "MTT S4000" \
  --driver <driver-version> \
  --toolkit <toolkit-version> \
  --mudnn <mudnn-version> \
  --mublas <mublas-version> \
  --image-digest sha256:<digest> \
  --build-flag=--cuda-gpu-arch=mp_22

python musa_operator_eval/tools/probe_sdpa.py \
  --output runs/sdpa_capability/torch_musa_sdpa.json
```

S5000 环境将架构和编译参数改为 `mp_31`，其他流程保持相同。

## 下一阶段

冻结 `sdpa_forward` 语义契约，实现 CPU FP64 reference、确定性张量生成、公开 smoke Case 和裸二进制 manifest。该阶段不依赖 MUSA 硬件。
