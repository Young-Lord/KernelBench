// The frozen half of the compiled starter's ABI (§4.6).
//
// Two things live here and nothing else: a reader for the tensor manifest, and
// the tensor type the kernel is handed. It is frozen because the evaluator writes
// the inputs and reads the outputs with its own copy of the same layout; a change
// here that the evaluator does not make is a change that silently reinterprets
// every tensor.
//
// The manifest is JSON, so this file contains a small JSON reader. It is written
// out rather than pulled in because a starter that needs a package manager is a
// starter that cannot be built on the machine the task is graded on, and the
// document it parses is one this repository generates.
#ifndef ABI_IO_H
#define ABI_IO_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// A compiler that can build both sides of this header defines its own qualifiers
// for "callable from the device"; a plain host compiler does not, and needs no
// qualifier at all. The helpers below are pure, so both sides can have them.
#if defined(__MUSACC__) || defined(__CUDACC__)
#define ABI_DEVICE_CALLABLE __host__ __device__
#else
#define ABI_DEVICE_CALLABLE
#endif

namespace abi {

// ---------------------------------------------------------------- JSON reader

struct Value {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<Value> items;
    std::map<std::string, Value> fields;

    const Value* find(const std::string& key) const {
        std::map<std::string, Value>::const_iterator it = fields.find(key);
        return it == fields.end() ? 0 : &it->second;
    }
    std::string string_or(const std::string& key, const std::string& fallback) const {
        const Value* value = find(key);
        return (value && value->kind == String) ? value->text : fallback;
    }
    double number_or(const std::string& key, double fallback) const {
        const Value* value = find(key);
        return (value && value->kind == Number) ? value->number : fallback;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text), position_(0) {}

    Value parse() {
        skip_space();
        Value value = parse_value();
        return value;
    }

private:
    const std::string& text_;
    size_t position_;

    void skip_space() {
        while (position_ < text_.size()) {
            char c = text_[position_];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { ++position_; continue; }
            break;
        }
    }
    bool eat(char expected) {
        skip_space();
        if (position_ < text_.size() && text_[position_] == expected) { ++position_; return true; }
        return false;
    }
    void expect(char expected) {
        if (!eat(expected)) {
            throw std::string("the tensor manifest is not the document this reader expects");
        }
    }
    Value parse_value() {
        skip_space();
        if (position_ >= text_.size()) throw std::string("the tensor manifest ends early");
        char c = text_[position_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') { Value value; value.kind = Value::String; value.text = parse_string(); return value; }
        if (!std::strncmp(text_.c_str() + position_, "true", 4)) { position_ += 4; Value v; v.kind = Value::Bool; v.boolean = true; return v; }
        if (!std::strncmp(text_.c_str() + position_, "false", 5)) { position_ += 5; Value v; v.kind = Value::Bool; v.boolean = false; return v; }
        if (!std::strncmp(text_.c_str() + position_, "null", 4)) { position_ += 4; return Value(); }
        return parse_number();
    }
    Value parse_object() {
        Value value; value.kind = Value::Object;
        expect('{');
        skip_space();
        if (eat('}')) return value;
        while (true) {
            std::string key = parse_string();
            expect(':');
            value.fields[key] = parse_value();
            if (eat(',')) continue;
            expect('}');
            break;
        }
        return value;
    }
    Value parse_array() {
        Value value; value.kind = Value::Array;
        expect('[');
        skip_space();
        if (eat(']')) return value;
        while (true) {
            value.items.push_back(parse_value());
            if (eat(',')) continue;
            expect(']');
            break;
        }
        return value;
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (position_ < text_.size()) {
            char c = text_[position_++];
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            if (position_ >= text_.size()) break;
            char escape = text_[position_++];
            switch (escape) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'u': {
                    // The manifests this repository writes are ASCII; a \u escape
                    // is decoded only far enough to not corrupt the stream.
                    if (position_ + 4 <= text_.size()) position_ += 4;
                    out.push_back('?');
                    break;
                }
                default: out.push_back(escape); break;
            }
        }
        throw std::string("a string in the tensor manifest is not terminated");
    }
    Value parse_number() {
        size_t start = position_;
        while (position_ < text_.size()) {
            char c = text_[position_];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') { ++position_; continue; }
            break;
        }
        Value value; value.kind = Value::Number;
        value.number = std::atof(text_.substr(start, position_ - start).c_str());
        return value;
    }
};

// ------------------------------------------------------------------- tensors

enum Dtype { F32 = 0, F16 = 1, BF16 = 2, U8 = 3 };

ABI_DEVICE_CALLABLE inline size_t element_size(Dtype dtype) {
    switch (dtype) {
        case F32: return 4;
        case F16: return 2;
        case BF16: return 2;
        default: return 1;
    }
}
ABI_DEVICE_CALLABLE inline Dtype dtype_from_name(const std::string& name) {
    if (name == "float32") return F32;
    if (name == "float16") return F16;
    if (name == "bfloat16") return BF16;
    return U8;
}
ABI_DEVICE_CALLABLE inline const char* dtype_name(Dtype dtype) {
    switch (dtype) {
        case F32: return "float32";
        case F16: return "float16";
        case BF16: return "bfloat16";
        default: return "uint8";
    }
}

// One tensor, held on the host as raw little-endian bytes. The kernel copies it
// to the device itself: keeping the transfer on the kernel's side is what lets the
// ABI stay free of both libtorch and the MUSA runtime.
struct Tensor {
    std::string name;
    Dtype dtype = F32;
    std::vector<long> shape;
    std::vector<unsigned char> bytes;

    size_t count() const {
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); ++i) total *= static_cast<size_t>(shape[i]);
        return total;
    }
    size_t expected_bytes() const { return count() * element_size(dtype); }
};

inline std::string join_path(const std::string& dir, const std::string& file) {
    if (!dir.empty() && dir[dir.size() - 1] == '/') return dir + file;
    return dir + "/" + file;
}

inline std::vector<Tensor> read_tensors(const std::string& directory) {
    std::ifstream manifest(join_path(directory, "tensors.json").c_str());
    if (!manifest.good()) throw std::string("no tensors.json in " + directory);
    std::stringstream buffer;
    buffer << manifest.rdbuf();

    Value document = Parser(buffer.str()).parse();
    const Value* list = document.find("tensors");
    if (!list || list->kind != Value::Array) throw std::string("the manifest has no tensor list");

    std::vector<Tensor> tensors;
    for (size_t index = 0; index < list->items.size(); ++index) {
        const Value& record = list->items[index];
        Tensor tensor;
        tensor.name = record.string_or("name", "");
        tensor.dtype = dtype_from_name(record.string_or("dtype", "float32"));
        const Value* shape = record.find("shape");
        if (shape && shape->kind == Value::Array) {
            for (size_t axis = 0; axis < shape->items.size(); ++axis) {
                tensor.shape.push_back(static_cast<long>(shape->items[axis].number));
            }
        }
        std::ifstream blob(join_path(directory, record.string_or("file", "")).c_str(), std::ios::binary);
        if (!blob.good()) throw std::string("missing tensor file " + record.string_or("file", ""));
        tensor.bytes.assign(std::istreambuf_iterator<char>(blob), std::istreambuf_iterator<char>());
        if (tensor.bytes.size() != tensor.expected_bytes()) {
            throw std::string("tensor " + tensor.name + " is " + std::to_string(tensor.bytes.size()) +
                              " bytes, its shape and dtype need " + std::to_string(tensor.expected_bytes()));
        }
        tensors.push_back(tensor);
    }
    return tensors;
}

// Writes the same shape of manifest the reader accepts. The digest and the byte
// count are recomputed here rather than copied, because the file this writes is
// what the evaluator compares against the golden.
inline void write_tensors(const std::string& directory, const std::vector<Tensor>& tensors) {
    std::string manifest = "{\n  \"schema_version\": \"1.0.0\",\n  \"reference\": \"compiled_runner\",\n  \"tensors\": [\n";
    for (size_t index = 0; index < tensors.size(); ++index) {
        const Tensor& tensor = tensors[index];
        std::string file = tensor.name + ".bin";
        std::ofstream blob(join_path(directory, file).c_str(), std::ios::binary);
        blob.write(reinterpret_cast<const char*>(tensor.bytes.empty() ? 0 : &tensor.bytes[0]),
                   static_cast<std::streamsize>(tensor.bytes.size()));
        blob.close();

        manifest += "    {\n      \"name\": \"" + tensor.name + "\",\n";
        manifest += "      \"file\": \"" + file + "\",\n";
        manifest += std::string("      \"dtype\": \"") + dtype_name(tensor.dtype) + "\",\n";
        manifest += "      \"shape\": [";
        for (size_t axis = 0; axis < tensor.shape.size(); ++axis) {
            manifest += (axis ? ", " : "") + std::to_string(tensor.shape[axis]);
        }
        manifest += "],\n";
        manifest += "      \"nbytes\": " + std::to_string(tensor.bytes.size()) + "\n";
        manifest += std::string("    }") + (index + 1 == tensors.size() ? "\n" : ",\n");
    }
    manifest += "  ]\n}\n";
    std::ofstream out(join_path(directory, "tensors.json").c_str());
    out << manifest;
}

}  // namespace abi

#endif  // ABI_IO_H
