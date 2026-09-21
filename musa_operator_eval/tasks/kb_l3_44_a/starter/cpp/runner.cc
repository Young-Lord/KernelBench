// The compiled starter's runner: input directory in, output directory out.
//
// §4.6 freezes this ABI: read the tensor manifest and its `.bin` files from the
// input directory, run the submission, write the manifest and its `.bin` files to
// the output directory, and do it without libtorch, so ATen cannot be reached
// from the process the submission runs in. This file and `abi_io.h` are the
// frozen half; `kernel.mu` is the editable one.
//
// The kernel fills the outputs completely -- name, dtype, shape and bytes --
// because what a task produces is the task's business, and a runner that knew the
// output shapes would be a second place for them to be written down.
//
// `--warmup W --repeat N` turns the run into a measurement: the inputs are read
// once, `kernel_entry` is called W times to bring the process and the device to a
// steady state (the first call in a process pays for the device context and for
// mapping a library the size of muDNN), and the next N calls are timed. Without
// the flags the behaviour is exactly one call and one write, which is what a caller
// that only wants an answer should ask for. The clock lives here rather than in the
// submission for the obvious reason: a submission that reported its own runtime
// would be reporting its own grade.
#include <musa_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "abi_io.h"

// The editable half implements this: it receives the inputs as they were read
// from disk, fills `outputs`, and returns zero on success.
extern "C" int kernel_entry(const std::vector<abi::Tensor>& inputs, std::vector<abi::Tensor>& outputs);

namespace {

double milliseconds(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

void usage(const char* program) {
    std::fprintf(stderr, "usage: %s <input-dir> <output-dir> [--warmup N] [--repeat N]\n", program);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    const std::string input_dir = argv[1];
    const std::string output_dir = argv[2];
    int warmup = 0;
    int repeat = 1;
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if ((flag == "--warmup" || flag == "--repeat") && i + 1 < argc) {
            const int value = std::atoi(argv[++i]);
            if (value < 0 || (flag == "--repeat" && value < 1)) {
                usage(argv[0]);
                return 2;
            }
            if (flag == "--warmup") warmup = value;
            else repeat = value;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    try {
        std::vector<abi::Tensor> inputs = abi::read_tensors(input_dir);
        std::vector<abi::Tensor> outputs;
        std::vector<double> timings;
        timings.reserve(static_cast<size_t>(repeat));
        for (int call = 0; call < warmup + repeat; ++call) {
            // The submission fills the vector completely on every call, so it starts
            // empty each time rather than accumulating outputs across calls.
            outputs.clear();
            // The previous call's device work is finished before the clock starts,
            // so a call is timed for its own work rather than for whatever the call
            // before it left running.
            musaDeviceSynchronize();
            const auto started = std::chrono::steady_clock::now();
            const int status = kernel_entry(inputs, outputs);
            musaDeviceSynchronize();
            const double elapsed = milliseconds(started, std::chrono::steady_clock::now());
            if (status != 0) {
                std::fprintf(stderr, "kernel_entry returned %d\n", status);
                return 3;
            }
            if (outputs.empty()) {
                std::fprintf(stderr, "kernel_entry produced no output\n");
                return 3;
            }
            if (call >= warmup) timings.push_back(elapsed);
        }
        abi::write_tensors(output_dir, outputs);
        // One line on stdout, and the statistic is the caller's to compute: this
        // reports what each timed call took, not what a grade should be.
        std::printf("{\"warmup\": %d, \"repeat\": %d, \"per_call_ms\": [", warmup, repeat);
        for (size_t i = 0; i < timings.size(); ++i) {
            std::printf("%s%.6f", i ? ", " : "", timings[i]);
        }
        std::printf("]}\n");
    } catch (const std::string& error) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 4;
    }
    return 0;
}
