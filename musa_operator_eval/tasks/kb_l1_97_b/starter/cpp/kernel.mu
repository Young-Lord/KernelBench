// The editable half of the compiled starter: the probe, the dispatch, the
// fallback and the trace, in C++.
//
// Fill the marked regions. The frozen half -- `abi_io.h`, `runner.cc` and
// `build.sh` -- reads the input directory, calls `kernel_entry`, and writes the
// output directory. `build.sh --extra-library <name>` is where the contract's
// whitelist is applied: a submission that links something else fails the link
// check, because the check reads this binary's dynamic section rather than its
// source.
#include <musa_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "abi_io.h"

// --- BEGIN dispatch_trace ---
// One record per case, appended to the file named by the environment variable in
// the contract's `trace_env_var`, with the fields it requires. A record that is
// missing, or that names a path other than the case's expected one, fails the
// run: the trace is what says which path earned the number.
static void record_dispatch(const char* path) {
    const char* trace_path = std::getenv("KB_DISPATCH_TRACE");
    const char* case_id = std::getenv("KB_DISPATCH_CASE_ID");
    if (!trace_path || !case_id) return;
    std::ofstream trace(trace_path, std::ios::app);
    trace << "{\"case_id\": \"" << case_id << "\", \"selected_path\": \"" << path
          << "\", \"probe_status\": \"accepted\"}\n";
}
// --- END dispatch_trace ---

// --- BEGIN probe_and_dispatch ---
// Probe the whitelisted libraries with a real call and read what comes back:
// shape, dtype and layout acceptance, and which attribute combinations are
// supported. Do not key the decision on a version string.
static const char* choose_path(const std::vector<abi::Tensor>& inputs) {
    (void)inputs;
    return "custom_fallback";
}
// --- END probe_and_dispatch ---

// --- BEGIN custom_fallback ---
__global__ void fallback_placeholder(const float* in, float* out, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) out[index] = in[index];
}

static int run_fallback(const std::vector<abi::Tensor>& inputs, std::vector<abi::Tensor>& outputs) {
    const abi::Tensor& source = inputs[0];
    abi::Tensor result;
    result.name = "output";
    result.dtype = source.dtype;
    result.shape = source.shape;
    result.bytes.resize(source.bytes.size());
    float* device_in = 0;
    float* device_out = 0;
    musaMalloc(reinterpret_cast<void**>(&device_in), source.bytes.size());
    musaMalloc(reinterpret_cast<void**>(&device_out), source.bytes.size());
    musaMemcpy(device_in, &source.bytes[0], source.bytes.size(), musaMemcpyHostToDevice);
    int count = static_cast<int>(source.count());
    fallback_placeholder<<<(count + 255) / 256, 256>>>(device_in, device_out, count);
    if (musaDeviceSynchronize() != musaSuccess) return 2;
    musaMemcpy(&result.bytes[0], device_out, source.bytes.size(), musaMemcpyDeviceToHost);
    musaFree(device_in); musaFree(device_out);
    outputs.push_back(result);
    return 0;
}
// --- END custom_fallback ---

extern "C" int kernel_entry(const std::vector<abi::Tensor>& inputs, std::vector<abi::Tensor>& outputs) {
    if (inputs.empty()) return 1;
    // --- BEGIN host_launch ---
    // Launch configuration for whichever path the probe selected.
    const char* path = choose_path(inputs);
    record_dispatch(path);
    if (std::string(path) != "custom_fallback") return 1;
    return run_fallback(inputs, outputs);
    // --- END host_launch ---
}
