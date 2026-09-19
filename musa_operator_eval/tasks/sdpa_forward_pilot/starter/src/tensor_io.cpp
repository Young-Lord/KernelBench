#include "tensor_io.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace musa_eval {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256Constants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::uint32_t rotate_right(std::uint32_t value, int amount) {
  return (value >> amount) | (value << (32 - amount));
}

std::string join_path(const std::string& directory, const std::string& file) {
  if (directory.empty() || directory.back() == '/' || directory.back() == '\\') return directory + file;
  return directory + "/" + file;
}

std::string read_text(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

std::string capture_string(const std::string& object, const std::string& key) {
  const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(object, match, expression)) throw std::runtime_error("missing string field " + key);
  return match[1].str();
}

std::uint64_t capture_uint(const std::string& object, const std::string& key) {
  const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(object, match, expression)) throw std::runtime_error("missing integer field " + key);
  return std::stoull(match[1].str());
}

std::vector<std::int64_t> capture_shape(const std::string& object) {
  const std::regex expression("\\\"shape\\\"\\s*:\\s*\\[([^\\]]+)\\]");
  std::smatch match;
  if (!std::regex_search(object, match, expression)) throw std::runtime_error("missing shape");
  std::vector<std::int64_t> shape;
  std::stringstream values(match[1].str());
  while (values) {
    std::int64_t value;
    values >> value;
    if (values.fail()) break;
    shape.push_back(value);
    if (values.peek() == ',') values.ignore();
  }
  if (shape.empty()) throw std::runtime_error("empty shape");
  return shape;
}

std::vector<std::string> tensor_objects(const std::string& document) {
  const auto tensor_key = document.find("\"tensors\"");
  const auto array_start = document.find('[', tensor_key);
  if (tensor_key == std::string::npos || array_start == std::string::npos) throw std::runtime_error("missing tensors array");
  std::vector<std::string> objects;
  int depth = 0;
  std::size_t start = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t index = array_start + 1; index < document.size(); ++index) {
    const char ch = document[index];
    if (in_string) {
      if (escaped) escaped = false;
      else if (ch == '\\') escaped = true;
      else if (ch == '"') in_string = false;
      continue;
    }
    if (ch == '"') in_string = true;
    else if (ch == '{') {
      if (depth++ == 0) start = index;
    } else if (ch == '}') {
      if (--depth == 0) objects.push_back(document.substr(start, index - start + 1));
    } else if (ch == ']' && depth == 0) {
      break;
    }
  }
  return objects;
}

std::string escape_json(const std::string& input) {
  std::string output;
  for (const char ch : input) {
    if (ch == '"' || ch == '\\') output.push_back('\\');
    output.push_back(ch);
  }
  return output;
}

}  // namespace

std::string sha256_hex(const std::vector<std::uint8_t>& input) {
  std::vector<std::uint8_t> message = input;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8;
  message.push_back(0x80);
  while ((message.size() % 64) != 56) message.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8) message.push_back(static_cast<std::uint8_t>(bit_length >> shift));
  std::array<std::uint32_t, 8> hash = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  for (std::size_t offset = 0; offset < message.size(); offset += 64) {
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      const auto base = offset + index * 4;
      words[index] = (message[base] << 24) | (message[base + 1] << 16) | (message[base + 2] << 8) | message[base + 3];
    }
    for (int index = 16; index < 64; ++index) {
      const auto s0 = rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
      const auto s1 = rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    auto a = hash[0], b = hash[1], c = hash[2], d = hash[3], e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    for (int index = 0; index < 64; ++index) {
      const auto sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const auto choice = (e & f) ^ (~e & g);
      const auto temp1 = h + sum1 + choice + kSha256Constants[index] + words[index];
      const auto sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = sum0 + majority;
      h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
  }
  std::ostringstream output;
  for (const auto value : hash) output << std::hex << std::setw(8) << std::setfill('0') << value;
  return output.str();
}

std::vector<Tensor> read_tensors(const std::string& directory) {
  const auto document = read_text(join_path(directory, "tensors.json"));
  std::vector<Tensor> tensors;
  for (const auto& object : tensor_objects(document)) {
    Tensor tensor;
    tensor.name = capture_string(object, "name");
    tensor.file = capture_string(object, "file");
    tensor.dtype = capture_string(object, "dtype");
    tensor.shape = capture_shape(object);
    tensor.layout = capture_string(object, "layout");
    const auto expected_bytes = capture_uint(object, "nbytes");
    const auto expected_sha = capture_string(object, "sha256");
    std::ifstream stream(join_path(directory, tensor.file), std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open tensor " + tensor.file);
    tensor.bytes.assign(std::istreambuf_iterator<char>(stream), {});
    if (tensor.bytes.size() != expected_bytes) throw std::runtime_error("nbytes mismatch for " + tensor.name);
    if (sha256_hex(tensor.bytes) != expected_sha) throw std::runtime_error("sha256 mismatch for " + tensor.name);
    tensors.push_back(std::move(tensor));
  }
  if (tensors.empty()) throw std::runtime_error("manifest contains no tensors");
  return tensors;
}

void write_tensors(const std::string& directory, const std::string& case_id, const std::vector<Tensor>& tensors) {
  std::ofstream manifest(join_path(directory, "tensors.json"), std::ios::binary);
  if (!manifest) throw std::runtime_error("cannot create output manifest");
  manifest << "{\n  \"schema_version\": \"1.0.0\",\n  \"case_id\": \"" << escape_json(case_id) << "\",\n  \"tensors\": [\n";
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto& tensor = tensors[index];
    const auto filename = tensor.file.empty() ? tensor.name + ".bin" : tensor.file;
    std::ofstream binary(join_path(directory, filename), std::ios::binary);
    binary.write(reinterpret_cast<const char*>(tensor.bytes.data()), static_cast<std::streamsize>(tensor.bytes.size()));
    if (!binary) throw std::runtime_error("failed to write tensor " + tensor.name);
    manifest << "    {\"name\": \"" << escape_json(tensor.name) << "\", \"file\": \"" << escape_json(filename)
             << "\", \"dtype\": \"" << tensor.dtype << "\", \"shape\": [";
    for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
      if (axis) manifest << ", ";
      manifest << tensor.shape[axis];
    }
    manifest << "], \"layout\": \"" << tensor.layout << "\", \"byte_order\": \"little\", \"nbytes\": " << tensor.bytes.size()
             << ", \"sha256\": \"" << sha256_hex(tensor.bytes) << "\"}" << (index + 1 == tensors.size() ? "\n" : ",\n");
  }
  manifest << "  ]\n}\n";
}

}  // namespace musa_eval
