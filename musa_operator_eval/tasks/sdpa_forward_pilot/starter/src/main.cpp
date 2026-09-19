#include "runner_api.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "usage: runner <input-dir> <output-dir> <case-id> <trace.jsonl> <causal:0|1> <window-left> <window-right> <scale> <reserved>\n";
    return 64;
  }
  try {
    musa_eval::RunAttributes attributes{argv[3], std::stoi(argv[5]) != 0, std::stoi(argv[6]), std::stoi(argv[7]), std::stof(argv[8])};
    const auto inputs = musa_eval::read_tensors(argv[1]);
    const auto decision = musa_eval::choose_dispatch(inputs, attributes);
    const auto outputs = musa_eval::run_sdpa(inputs, attributes, decision);
    musa_eval::write_tensors(argv[2], attributes.case_id, outputs);
    musa_eval::append_dispatch_trace(argv[4], attributes.case_id, decision);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 65;
  }
}
