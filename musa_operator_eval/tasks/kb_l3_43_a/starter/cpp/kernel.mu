// The editable half of the compiled starter: the device kernel and its launch.
//
// Fill the marked regions. The frozen half -- `abi_io.h`, `runner.cc` and
// `build.sh` -- reads the input directory, calls `kernel_entry`, and writes the
// output directory; nothing here has to know about files.
#include <musa_runtime.h>

#include <cmath>
#include <string>
#include <vector>

#include "abi_io.h"

// --- BEGIN device_kernel ---
// The core computation. The A tier requires all of it to happen here: no muDNN,
// no muBLAS, no SDPA, no ATen. `to_float` / `put_float` handle the storage dtypes
// the case declares (float32 for this package's cases; float16 and bfloat16 are
// implemented beside them if you need them).
__global__ void kernel_placeholder(const float* in, float* out, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) out[index] = in[index];
}
// --- END device_kernel ---

extern "C" int kernel_entry(const std::vector<abi::Tensor>& inputs, std::vector<abi::Tensor>& outputs) {
    if (inputs.empty()) return 1;
    const abi::Tensor& source = inputs[0];

    // --- BEGIN host_launch ---
    // grid / block / shared memory, and the transfers around the launch. The
    // placeholder below copies the first input through so that the starter builds
    // and runs before anything is filled in; replace it.
    abi::Tensor result;
    result.name = "output";
    result.dtype = source.dtype;
    result.shape = source.shape;
    result.bytes.resize(source.bytes.size());
    if (source.dtype != abi::F32) return 1;

    float* device_in = 0;
    float* device_out = 0;
    musaMalloc(reinterpret_cast<void**>(&device_in), source.bytes.size());
    musaMalloc(reinterpret_cast<void**>(&device_out), source.bytes.size());
    musaMemcpy(device_in, &source.bytes[0], source.bytes.size(), musaMemcpyHostToDevice);
    int count = static_cast<int>(source.count());
    kernel_placeholder<<<(count + 255) / 256, 256>>>(device_in, device_out, count);
    if (musaDeviceSynchronize() != musaSuccess) return 2;
    musaMemcpy(&result.bytes[0], device_out, source.bytes.size(), musaMemcpyDeviceToHost);
    musaFree(device_in);
    musaFree(device_out);
    // --- END host_launch ---

    outputs.push_back(result);
    return 0;
}
