"""SDPA forward library dispatch —— B 类:库集成 Starter。

把本文件复制成 `model_new.py`,填好标记区域后提交。本包内除 `starter/` 之外的文件都是
冻结的,清单见 `task.json` 的 `starter.frozen`;改冻结点会被静态审计按越界修改拒绝,
这份副本里的其他内容可以自由改写。

接口:`ModelNew(scale, causal, window_left, window_right)`,`forward(q, k, v)` → 一个与 reference 输出同形状的张量。
输入:q/k/v 三个 BHSD 张量。

    # 生成公开 smoke 用例的输入与 golden(不提交,按需生成)
    python musa_operator_eval/tools/generate_cases.py --task-dir musa_operator_eval/tasks/sdpa_forward_pilot
    # 不需要设备的自测:静态审计 + case 覆盖检查 + golden 交叉核对
    python musa_operator_eval/evaluator/run_task.py --task-dir musa_operator_eval/tasks/sdpa_forward_pilot \
        --submission model_new.py --static-only
    # 完整评测:静态审计 → 编译 → 链接白名单核对 → 正确性 → 性能
    python musa_operator_eval/evaluator/run_task.py --task-dir musa_operator_eval/tasks/sdpa_forward_pilot \
        --submission model_new.py --backend musa --precision fp16

输入与输出只在两个目录之间以裸二进制张量加 JSON manifest 传递,格式见
`task.json` 的 `starter.abi`。提交物本身是一个定义 `ModelNew` 的 Python 模块:
评测端在与提交物同一个进程里运行 reference,这也是本包对 §4.6 "runner 不引入
libtorch" 的偏差所在,记在 `starter.abi.runner_imports_libtorch` 里而不是藏起来。
"""

import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """SDPA forward library dispatch:填好下面四个标记区域。"""

    def __init__(self, scale, causal, window_left, window_right):
        super().__init__()
        # --- BEGIN state ---
        # 与 reference 同序、同名地构建同样的参数;评测端用同一个 seed 分别重建
        # reference 与提交物,靠构造过程对齐权重,不复制 state_dict。
        # --- END state ---

    def forward(self, q, k, v):
        # --- BEGIN probe_and_dispatch ---
        # 先用运行时返回值探测白名单库能不能一次调用给出完全相同的语义
        # (形状 / dtype / layout 是否接受、属性组合是否支持),再据此选路径。
        # 禁止按版本号硬编码分派。库调用只允许出现在这一段与下面两个区域里。
        # --- END probe_and_dispatch ---

        # --- BEGIN host_launch ---
        # 启动配置,以及库路径之外的 Host 端编排(中间张量的分配与释放)。
        # --- END host_launch ---

        # --- BEGIN custom_fallback ---
        # 库路径兜不住的配置才走这里,可以 JIT 编译 .mu kernel。
        # 不要把它当成全局降级:每个 case 期望的路径都会与分派轨迹逐条核对。
        # --- END custom_fallback ---

        # --- BEGIN dispatch_trace ---
        # 每个 case 追加一条分派记录,字段见 library_policy.required_trace_fields,
        # 轨迹文件与 case id 由环境变量给出(见 library_policy.trace_env_var /
        # case_id_env_var)。记录缺字段或不匹配期望路径都不算通过。
        # --- END dispatch_trace ---
        raise NotImplementedError("fill the marked regions above")
