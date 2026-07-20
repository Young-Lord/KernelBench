# GPU 并行计算 (MUSA Programming Guide)

> Source: https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/programming_guide/what_is_musa/gpu_parallel_basics
> Fetched for KernelBench MUSA adaptation reference.

## GPU 概述

图形处理器 GPU（Graphics Processing Unit）是一种专为并行计算设计的处理器。GPU 最初用于加速计算机图形渲染，由于其高吞吐量的计算特性，现已广泛应用于通用并行计算 GPGPU（General-Purpose GPU Computing）领域。

### GPU 与 CPU 的架构差异

| 特性 | CPU | GPU |
| --- | --- | --- |
| 设计目标 | 低延迟，优化单线程性能 | 高吞吐量，优化并行处理能力 |
| 核数 | 较少（通常 4-64 核） | 大量（数千个计算单元） |
| 缓存设计 | 大容量缓存，复杂控制逻辑 | 较小缓存，更多计算单元 |
| 适用场景 | 串行任务、复杂分支逻辑 | 大规模数据并行计算 |

以 Intel Xeon 8280 和 MTT S5000 为例：

| 指标 | CPU (Intel Xeon 8280) | GPU (MTT S5000) |
| --- | --- | --- |
| 核数 | 28 核 | 4,096 核 |
| 时钟频率 | 2.7 GHz | 1.8 GHz |
| 内存带宽 | 140 GB/s | 448 GB/s |
| 并发线程数 | ~1,792 | ~196,608 |
| 单精度浮点性能 | 4.8 TFLOPS | 14.7 TFLOPS |

## GPU 并行计算模型

### 异构计算架构

MUSA 采用异构计算模型：CPU 作为主机（Host），GPU 作为设备（Device）。Host 和 Device 拥有独立的内存空间，通过 PCIe 等总线进行数据传输。

典型执行流程：
- 主机（CPU）：负责顺序逻辑、内存管理、内核（Kernel）启动
- 设备（GPU）：负责大规模并行计算
- 数据传输：通过 `musaMemcpy()` 等 API 在主机和设备间拷贝数据

### 线程层次结构

MUSA 采用三层线程层次结构组织并行计算：

- Grid（网格）：一次 Kernel 启动的所有线程
- Block（线程块）：线程组，可独立调度执行
- Thread（线程）：单个执行单元

每个线程通过内置变量获取其在层次结构中的位置：

```cpp
int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
```

| 内置变量 | 说明 |
| --- | --- |
| `threadIdx` | 线程在 Block 内的索引 |
| `blockIdx` | Block 在 Grid 内的索引 |
| `blockDim` | Block 的维度大小 |
| `gridDim` | Grid 的维度大小 |

### SIMT 执行模型

GPU 采用单指令多线程 SIMT（Single Instruction Multiple Threads）执行模型：

- 一组线程（Warp）同时执行同一条指令
- 每个线程具有独立的寄存器状态
- 当线程执行路径出现分支时，Warp 序列化执行各分支

Warp 大小：不同 GPU 架构的 Warp 大小不同

- MTT M1000/S4000：**Warp = 128 线程**
- MTT S5000：Warp = 32 线程

选择 Block 大小时，建议使用 Warp 大小的倍数（如 128、256、512、1024）。

## GPU 内存层次结构

| 内存类型 | 访问速度 | 容量 | 可见范围 |
| --- | --- | --- | --- |
| 寄存器（Register） | 最快（1 周期） | 每线程有限 | 线程私有 |
| 共享内存（Shared Memory） | 快（1-2 周期） | 每 Block 有限 | Block 内共享 |
| 常量内存（Constant Memory） | 中（缓存命中时快） | 较小 | 全局可见 |
| 全局内存（Global Memory） | 慢（100+ 周期） | 大 | 全局可见 |

## MUSA 编程模型核心抽象

### Kernel 函数

Kernel 是在 GPU 上执行的函数，使用 `__global__` 修饰符声明：

```cpp
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Kernel 启动

```cpp
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

### 内存管理

| API | 说明 |
| --- | --- |
| `musaMalloc(&ptr, size)` | 分配设备内存 |
| `musaMemcpy(dst, src, size, kind)` | 在 Host 和 Device 间拷贝数据 |
| `musaFree(ptr)` | 释放设备内存 |

### 线程同步

```cpp
__syncthreads();  // 等待 Block 内所有线程到达此同步点
```

## GPU 程序执行流程

1. Host 分配内存并初始化数据
2. Host 分配 Device 内存
3. Host 将数据拷贝到 Device
4. Host 启动 Kernel，在 Device 上并行执行
5. Device 完成计算
6. Host 将结果拷贝回 Host 内存
7. Host 验证结果并释放内存
