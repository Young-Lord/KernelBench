import torch
import torch.nn as nn
from kernelbench.musa_extension import load_inline

scan_source = r"""
#include <torch/extension.h>
#include <musa_runtime.h>
#include <cstdint>
#include <cfloat>
#include <vector>

// Shared scan kernel used by cumulative sum/product problems.
//   op:   0 = cumulative sum, 1 = cumulative product
//   mode: 0 = inclusive, 1 = exclusive (shifted), 2 = reverse (sum over suffix)
// An optional boolean mask zeroes out masked positions before scanning.
// Aggregates are kept in double to minimize accumulation error.
__global__ void kernelbench_scan_kernel(
    const float* input,
    float* output,
    const bool* mask,
    int64_t outer_size,
    int64_t scan_size,
    int64_t inner_size,
    int64_t op,
    int64_t mode) {
    const int64_t seq_index = static_cast<int64_t>(blockIdx.x);
    const int64_t outer_index = seq_index / inner_size;
    const int64_t inner_index = seq_index - outer_index * inner_size;
    const int64_t base_offset = (outer_index * scan_size) * inner_size + inner_index;
    const int64_t stride = inner_size;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int chunk_size = static_cast<int>((scan_size + block_size - 1) / block_size);
    const int64_t start = static_cast<int64_t>(tid) * chunk_size;
    const int64_t end = start + chunk_size < scan_size ? start + chunk_size : scan_size;

    extern __shared__ double shared_mem[];
    double* segment_values = shared_mem;
    double* segment_inclusive = shared_mem + block_size;
    double* segment_temp = shared_mem + 2 * block_size;

    const bool is_prod = (op == 1);

    // Pass 1: aggregate this thread's chunk.
    double local_agg = is_prod ? 1.0 : 0.0;
    if (mode == 2) {
        for (int64_t i = end - 1; i >= start; --i) {
            float v = input[base_offset + i * stride];
            if (mask != nullptr && !mask[base_offset + i * stride]) v = 0.0f;
            local_agg = is_prod ? local_agg * v : local_agg + v;
        }
    } else {
        for (int64_t i = start; i < end; ++i) {
            float v = input[base_offset + i * stride];
            if (mask != nullptr && !mask[base_offset + i * stride]) v = 0.0f;
            local_agg = is_prod ? local_agg * v : local_agg + v;
        }
    }
    segment_values[tid] = local_agg;
    __syncthreads();

    // Inclusive scan over chunk aggregates (Hillis-Steele with temp buffer
    // so each step reads the previous iteration's values, avoiding races).
    segment_inclusive[tid] = segment_values[tid];
    __syncthreads();
    for (int step = 1; step < block_size; step <<= 1) {
        if (tid >= step) {
            segment_temp[tid] = is_prod
                ? segment_inclusive[tid - step] * segment_inclusive[tid]
                : segment_inclusive[tid - step] + segment_inclusive[tid];
        } else {
            segment_temp[tid] = segment_inclusive[tid];
        }
        __syncthreads();
        segment_inclusive[tid] = segment_temp[tid];
        __syncthreads();
    }

    double prefix;
    if (mode == 2) {
        // Reverse mode: prefix is the sum of all chunks after this one.
        prefix = segment_inclusive[block_size - 1] - segment_inclusive[tid];
    } else if (is_prod) {
        // Exclusive prefix for this thread's chunk (product). A zero chunk
        // (a true 0.0 element or double underflow) would make the division
        // produce inf/nan, so fall back to the previous inclusive product.
        const double chunk_agg = segment_values[tid];
        if (chunk_agg != 0.0) {
            prefix = segment_inclusive[tid] / chunk_agg;
        } else if (tid == 0) {
            prefix = 1.0;
        } else {
            prefix = segment_inclusive[tid - 1];
        }
    } else {
        // Exclusive prefix for this thread's chunk (sum).
        prefix = segment_inclusive[tid] - segment_values[tid];
    }
    __syncthreads();

    // Pass 2: emit results using the prefix.
    double running = is_prod ? 1.0 : 0.0;
    if (mode == 2) {
        for (int64_t i = end - 1; i >= start; --i) {
            float v = input[base_offset + i * stride];
            if (mask != nullptr && !mask[base_offset + i * stride]) v = 0.0f;
            running = is_prod ? running * v : running + v;
            output[base_offset + i * stride] = static_cast<float>(
                is_prod ? prefix * running : prefix + running);
        }
    } else if (mode == 1) {
        for (int64_t i = start; i < end; ++i) {
            output[base_offset + i * stride] = static_cast<float>(
                is_prod ? prefix * running : prefix + running);
            float v = input[base_offset + i * stride];
            if (mask != nullptr && !mask[base_offset + i * stride]) v = 0.0f;
            running = is_prod ? running * v : running + v;
        }
    } else {
        for (int64_t i = start; i < end; ++i) {
            float v = input[base_offset + i * stride];
            if (mask != nullptr && !mask[base_offset + i * stride]) v = 0.0f;
            running = is_prod ? running * v : running + v;
            output[base_offset + i * stride] = static_cast<float>(
                is_prod ? prefix * running : prefix + running);
        }
    }
}

torch::Tensor kernelbench_scan(
    torch::Tensor input,
    c10::optional<torch::Tensor> mask,
    int64_t dimension,
    int64_t op,
    int64_t mode) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int64_t dim_count = input.dim();
    const int64_t normalized_dim = dimension < 0 ? dimension + dim_count : dimension;
    TORCH_CHECK(normalized_dim >= 0 && normalized_dim < dim_count, "dimension out of range");

    int64_t outer_size = 1;
    int64_t inner_size = 1;
    for (int64_t axis = 0; axis < normalized_dim; ++axis) outer_size *= input.size(axis);
    const int64_t scan_size = input.size(normalized_dim);
    for (int64_t axis = normalized_dim + 1; axis < dim_count; ++axis) inner_size *= input.size(axis);

    torch::Tensor output = torch::empty_like(input);

    const bool* mask_ptr = nullptr;
    if (mask.has_value() && mask->defined()) {
        TORCH_CHECK(mask->scalar_type() == torch::kBool, "mask must be bool");
        TORCH_CHECK(mask->is_contiguous(), "mask must be contiguous");
        TORCH_CHECK(mask->sizes() == input.sizes(), "mask shape must match input");
        mask_ptr = mask->data_ptr<bool>();
    }

    const int64_t sequence_count = outer_size * inner_size;
    if (sequence_count > 0 && scan_size > 0) {
        constexpr int threads_per_block = 256;
        TORCH_CHECK(sequence_count <= 2147483647LL, "grid is too large");
        const size_t shared_bytes = 3 * threads_per_block * sizeof(double);
        kernelbench_scan_kernel<<<static_cast<unsigned int>(sequence_count), threads_per_block, shared_bytes>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            mask_ptr,
            outer_size,
            scan_size,
            inner_size,
            op,
            mode);
    }
    return output;
}
"""

scan_extension = load_inline(
    name="kernelbench_level1_problem93_scan_musa",
    cpp_sources=scan_source,
    functions=["kernelbench_scan"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return scan_extension.kernelbench_scan(x, mask, self.dim, 0, 0)
