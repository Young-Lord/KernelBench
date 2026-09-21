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
//
// The measurement is the *framework's* protocol, not one of this file's own making,
// so that a compiled number and a baseline's `latency_ms` -- which
// `kernelbench.timing` produces -- are the same kind of number:
//
//   * the clock is a pair of MUSA events around the call, as upstream's
//     `time_execution_with_cuda_event` uses, not a host clock;
//   * the L2 cache is flushed before every timed call (a 256 MB fill, which is what
//     upstream's `clear_l2_cache` is), so the numbers are cold-cache numbers; the
//     fill is queued before the start event, so its own duration is outside the
//     measured interval;
//   * the first timed call is discarded, as `discard_first=1` does upstream;
//   * the samples are reported and the *statistic* is the caller's, because the
//     driver takes the mean the framework's `get_timing_stats` takes -- this file
//     reports what each call took, not what a grade should be.
//
// The one boundary worth knowing before reading a number from here: the events are
// recorded on the default stream, so a submission that launches its work on a stream
// of its own and does not synchronize will be timed as if it had done nothing. Every
// worked answer and every skeleton uses the default stream, which is why the ABI
// documents the submission's own allocations and copies as the pattern to follow.
#include <musa_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "abi_io.h"

// The editable half implements this: it receives the inputs as they were read
// from disk, fills `outputs`, and returns zero on success.
extern "C" int kernel_entry(const std::vector<abi::Tensor>& inputs, std::vector<abi::Tensor>& outputs);

namespace {

// Upstream's `clear_l2_cache` allocates a 32 x 1024 x 1024 int64 tensor and fills
// it: 256 MB written to capacity, which is the point of it.
constexpr size_t THRASH_BYTES = 256ull * 1024 * 1024;

void usage(const char* program) {
    std::fprintf(stderr, "usage: %s <input-dir> <output-dir> [--warmup N] [--repeat N]\n", program);
}

// A buffer whose only purpose is to evict what the previous call left in the cache.
// A failed allocation is not fatal: the run is still a run, it is a warm-cache one,
// and the reported `l2_thrash_bytes` says so rather than leaving it to be guessed.
size_t prepare_thrash_buffer(void** buffer) {
    if (musaMalloc(buffer, THRASH_BYTES) != musaSuccess) {
        *buffer = nullptr;
        return 0;
    }
    return THRASH_BYTES;
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

        void* thrash = nullptr;
        const size_t thrash_bytes = prepare_thrash_buffer(&thrash);
        musaEvent_t start = nullptr;
        musaEvent_t end = nullptr;
        const bool timed = repeat > 0;
        if (timed && (musaEventCreate(&start) != musaSuccess || musaEventCreate(&end) != musaSuccess)) {
            std::fprintf(stderr, "cannot create the timing events\n");
            return 5;
        }
        // The first timed call is a discarded one, so the reported samples are all
        // of a warmed cache: one extra call is enqueued and dropped. A single-call
        // run (`--repeat 1`, and the no-flags default) has nothing to warm, so it
        // stays one call and one write, which is what the header promises.
        const int discarded = repeat > 1 ? 1 : 0;
        const int calls = warmup + repeat + discarded;
        std::vector<double> timings;
        timings.reserve(static_cast<size_t>(repeat));
        for (int call = 0; call < calls; ++call) {
            // The submission fills the vector completely on every call, so it starts
            // empty each time rather than accumulating outputs across calls.
            outputs.clear();
            const bool measure_this_call = call >= warmup + discarded;
            double elapsed = 0.0;
            if (measure_this_call) {
                // The device is idle before the flush is queued, so the interval
                // that follows holds this call's work and the wait for the eviction
                // is stream-ordered behind it rather than inside it.
                musaDeviceSynchronize();
                if (thrash != nullptr) {
                    musaMemset(thrash, 42, thrash_bytes);
                }
                musaEventRecord(start, 0);
            }
            const int status = kernel_entry(inputs, outputs);
            if (measure_this_call) {
                musaEventRecord(end, 0);
                musaEventSynchronize(end);
                float milliseconds = 0.0f;
                if (musaEventElapsedTime(&milliseconds, start, end) == musaSuccess) {
                    elapsed = milliseconds;
                }
                timings.push_back(elapsed);
            } else {
                // A warmup call is not timed, but its device work has to be finished
                // before the next call starts, or the next one would be waiting for it.
                musaDeviceSynchronize();
            }
            if (status != 0) {
                std::fprintf(stderr, "kernel_entry returned %d\n", status);
                return 3;
            }
            if (outputs.empty()) {
                std::fprintf(stderr, "kernel_entry produced no output\n");
                return 3;
            }
        }
        // The output belongs to the last call, and the copy back has to have landed
        // before it is read out of the host-side tensor.
        musaDeviceSynchronize();
        abi::write_tensors(output_dir, outputs);
        if (thrash != nullptr) musaFree(thrash);
        if (timed) {
            musaEventDestroy(start);
            musaEventDestroy(end);
        }
        // One line on stdout, and the statistic is the caller's to compute: this
        // reports what each timed call took, not what a grade should be. The fields
        // that describe the protocol travel with the samples, so a record cannot
        // claim a cold-cache event measurement it did not make.
        std::printf(
            "{\"warmup\": %d, \"repeat\": %d, \"discarded\": %d, \"timer\": \"musa_event\", "
            "\"l2_thrash_bytes\": %zu, \"per_call_ms\": [",
            warmup, repeat, discarded, thrash_bytes);
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
