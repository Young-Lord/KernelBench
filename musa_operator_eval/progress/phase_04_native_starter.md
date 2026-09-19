# 阶段 04：Native Starter 与冻结边界

状态：已完成并通过 host-only 验证  
日期：2026-09-19

## 已交付

- 不依赖 libtorch/ATen 的 C++17 裸二进制 manifest IO。
- IO 读取时校验 `nbytes` 与 SHA-256，写出时重新计算 SHA-256。
- 冻结的 runner API、主程序和 IO；A/B 可共用这些文件。
- B 类 dispatcher、Host launch 和 fallback kernel 三个明确的 Agent 可编辑区域。
- JSON Lines 分派轨迹，包含 `case_id`、`selected_path`、`probe_status`。
- 唯一构建入口 `starter/build.py`，支持 host-only、`mp_22` 和 `mp_31`。
- Starter 文件权限清单和冻结文件哈希锁工具。
- host-only IO round-trip 单元测试。

## 尚需硬件验证

- `mcc` 对当前命令行和 `.mu` 多源构建的兼容性。
- muDNN 实际库名、链接目录及允许符号前缀。
- fused/library-composition adapter 的真实调用签名。
- fallback kernel 的实现与性能。

## 验证命令

```bash
python musa_operator_eval/tools/freeze_starter.py --write
python musa_operator_eval/tools/freeze_starter.py
python -m unittest discover -s musa_operator_eval/tests -v
```

验证结果：旧版 MinGW/中文绝对路径暴露的 `std::filesystem` 兼容问题已通过移除该依赖、统一相对路径解决；冻结哈希校验、C++17 编译、SHA-256 校验及输入输出往返均通过。前四阶段累计 12 项测试全部通过。

## 下一阶段

实现 evaluator 的 CPU 可测控制流：Schema、Starter 边界、禁用符号静态审计、构建、公开正确性、分派轨迹和报告短路逻辑。MUSA 执行与性能阶段先以明确的 `not_run` 适配器保留，不伪造通过状态。
