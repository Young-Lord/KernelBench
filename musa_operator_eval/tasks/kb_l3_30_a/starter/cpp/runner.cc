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
#include <cstdio>
#include <string>
#include <vector>

#include "abi_io.h"

// The editable half implements this: it receives the inputs as they were read
// from disk, fills `outputs`, and returns zero on success.
extern "C" int kernel_entry(const std::vector<abi::Tensor>& inputs, std::vector<abi::Tensor>& outputs);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <input-dir> <output-dir>\n", argv[0]);
        return 2;
    }
    try {
        std::vector<abi::Tensor> inputs = abi::read_tensors(argv[1]);
        std::vector<abi::Tensor> outputs;
        int status = kernel_entry(inputs, outputs);
        if (status != 0) {
            std::fprintf(stderr, "kernel_entry returned %d\n", status);
            return 3;
        }
        if (outputs.empty()) {
            std::fprintf(stderr, "kernel_entry produced no output\n");
            return 3;
        }
        abi::write_tensors(argv[2], outputs);
    } catch (const std::string& error) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 4;
    }
    return 0;
}
