"""Swin Transformer V2 —— A 类:算子实现 Starter。

把本文件复制成 `model_new.py`,填好标记区域后提交。本包内除 `starter/` 之外的文件都是
冻结的,清单见 `task.json` 的 `starter.frozen`;改冻结点会被静态审计按越界修改拒绝,
这份副本里的其他内容可以自由改写。

接口:`ModelNew()`,`forward(x)` → 一个与 reference 输出同形状的张量。
输入:x 一个 (B, 3, H, W) 张量。

    # 生成公开 smoke 用例的输入与 golden(不提交,按需生成)
    python musa_operator_eval/tools/generate_cases.py --task-dir musa_operator_eval/tasks/kb_l3_30_a
    # 不需要设备的自测:静态审计 + case 覆盖检查 + golden 交叉核对
    python musa_operator_eval/evaluator/run_task.py --task-dir musa_operator_eval/tasks/kb_l3_30_a \
        --submission model_new.py --static-only
    # 完整评测:静态审计 → 编译 → 链接白名单核对 → 正确性 → 性能
    python musa_operator_eval/evaluator/run_task.py --task-dir musa_operator_eval/tasks/kb_l3_30_a \
        --submission model_new.py --backend musa --precision fp32

输入与输出只在两个目录之间以裸二进制张量加 JSON manifest 传递,格式见
`task.json` 的 `starter.abi`。提交物本身是一个定义 `ModelNew` 的 Python 模块:
评测端在与提交物同一个进程里运行 reference,这也是本包对 §4.6 "runner 不引入
libtorch" 的偏差所在,记在 `starter.abi.runner_imports_libtorch` 里而不是藏起来。
"""

import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """Swin Transformer V2:填好下面三个标记区域。"""

    def __init__(self):
        super().__init__()
        # --- BEGIN state ---
        # 与 reference 同序、同名地构建同样的参数;评测端用同一个 seed 分别重建
        # reference 与提交物,靠构造过程对齐权重,不复制 state_dict。
        # --- END state ---

    def forward(self, x):
        # --- BEGIN device_kernel ---
        # 核心计算全部写在 Device kernel 里:kernel 源码(可以用
        # kernelbench.musa_extension.load_inline JIT 编译 .mu 源码)以及必要的
        # 中间张量。库算子、ATen、SDPA 都不能用来完成这一段。
        # --- END device_kernel ---

        # --- BEGIN host_launch ---
        # grid / block / shared memory 的配置与启动。
        # --- END host_launch ---
        raise NotImplementedError("fill the marked regions above")
