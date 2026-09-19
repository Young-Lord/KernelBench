#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace musa_eval {

struct Tensor {
  std::string name;
  std::string file;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::string layout;
  std::vector<std::uint8_t> bytes;
};

std::vector<Tensor> read_tensors(const std::string& directory);
void write_tensors(const std::string& directory,
                   const std::string& case_id,
                   const std::vector<Tensor>& tensors);
std::string sha256_hex(const std::vector<std::uint8_t>& bytes);

}  // namespace musa_eval
