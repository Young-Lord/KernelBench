# KernelBench MUSA (Moore Threads GPU) Support

This directory contains reference documentation and setup notes for running KernelBench on Moore Threads MUSA GPUs.

## Requirements

- Moore Threads GPU (tested on **MTT S4000**)
- [MUSA SDK](https://developer.mthreads.com/) (driver + toolkit + muDNN)
- [torch_musa](https://github.com/MooreThreads/Torch_MUSA) PyTorch backend

## Environment Setup

```bash
export MUSA_HOME=/usr/local/musa
export PATH=$MUSA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH

# Verify GPU is visible
musaInfo
```

Install KernelBench with GPU extras:

```bash
uv sync --extra gpu
```

## Running Evaluation on MUSA

Use `backend=musa` and set the GPU architecture for your card (e.g. `mp_22` for MTT S4000):

```bash
uv run python scripts/run_and_check.py \
  ref_origin=local \
  ref_arch_src_path=src/kernelbench/prompts/model_ex_add.py \
  kernel_src_path=src/kernelbench/prompts/model_new_ex_add_musa.py \
  eval_mode=local \
  backend=musa \
  gpu_arch='["mp_22"]'
```

Generate and evaluate a single sample:

```bash
uv run python scripts/generate_and_eval_single_sample.py \
  level=1 problem_id=1 backend=musa gpu_arch='["mp_22"]'
```

## Backend Overview

The `musa` backend follows the same pattern as `hip` (AMD):

- Custom kernels use `__global__` functions compiled with **mcc** (MUSA compiler)
- Inline compilation via `torch_musa.utils.musa_extension.MUSAExtension`
- Device API uses `torch.musa.*` (via KernelBench's GPU abstraction layer)
- Prompts include MTT S4000 hardware specs for LLM guidance

## Architecture Identifiers

| GPU | MUSA Arch | Warp Size |
| --- | --- | --- |
| MTT S4000 | `mp_22` | 128 |
| MTT S80 / S3000 | `mp_21` | 128 |
| MTT S5000 | `mp_31` | 32 |

Set via `gpu_arch` config or `TORCH_MUSA_ARCH` environment variable.

## Key Differences from CUDA

1. **Device type**: Use `musa:0` instead of `cuda:0` (KernelBench auto-resolves when platform is MUSA)
2. **Headers**: Use `<musa_runtime.h>` or CUDA-compatible `<cuda_runtime.h>` from torch_musa
3. **Compiler**: `mcc` instead of `nvcc`
4. **Extension API**: `MUSAExtension` + `BuildExtension` from `torch_musa.utils.musa_extension`

## Reference Docs

- [gpu_parallel_basics.md](./gpu_parallel_basics.md) — MUSA programming model overview
- [Moore Threads Documentation Center](https://docs.mthreads.com/)
- [Torch MUSA GitHub](https://github.com/MooreThreads/Torch_MUSA)
