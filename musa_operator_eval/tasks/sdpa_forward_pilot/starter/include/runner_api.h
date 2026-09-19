#pragma once

#include "tensor_io.h"

#include <string>
#include <vector>

namespace musa_eval {

enum class DispatchPath { kFusedLibrary, kLibraryComposition, kCustomFallback };

struct DispatchDecision {
  DispatchPath path;
  std::string probe_status;
};

struct RunAttributes {
  std::string case_id;
  bool causal;
  int window_left;
  int window_right;
  float scale;
};

DispatchDecision choose_dispatch(const std::vector<Tensor>& inputs,
                                 const RunAttributes& attributes);
std::vector<Tensor> run_sdpa(const std::vector<Tensor>& inputs,
                             const RunAttributes& attributes,
                             const DispatchDecision& decision);
void append_dispatch_trace(const std::string& path,
                           const std::string& case_id,
                           const DispatchDecision& decision);

}  // namespace musa_eval
