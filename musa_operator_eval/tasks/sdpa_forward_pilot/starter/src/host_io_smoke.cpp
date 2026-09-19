#include "tensor_io.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: host_io_smoke <input-dir> <output-dir> <case-id>\n";
    return 64;
  }
  try {
    const auto tensors = musa_eval::read_tensors(argv[1]);
    musa_eval::write_tensors(argv[2], argv[3], tensors);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 65;
  }
}
