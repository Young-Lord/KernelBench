#include "runner_api.h"

#include <fstream>
#include <stdexcept>

namespace musa_eval {

DispatchDecision choose_dispatch(const std::vector<Tensor>& inputs,
                                 const RunAttributes& attributes) {
  (void)inputs;
  (void)attributes;
  // BEGIN_AGENT_EDIT:DISPATCH
  // Replace this placeholder with runtime capability probes and shape-based
  // dispatch. Do not branch on library version strings.
  return {DispatchPath::kCustomFallback, "starter_placeholder"};
  // END_AGENT_EDIT:DISPATCH
}

void append_dispatch_trace(const std::string& path,
                           const std::string& case_id,
                           const DispatchDecision& decision) {
  const char* selected = decision.path == DispatchPath::kFusedLibrary
                             ? "fused_library"
                             : decision.path == DispatchPath::kLibraryComposition
                                   ? "library_composition"
                                   : "custom_fallback";
  std::ofstream output(path, std::ios::app);
  if (!output) throw std::runtime_error("cannot open dispatch trace");
  output << "{\"case_id\":\"" << case_id << "\",\"selected_path\":\""
         << selected << "\",\"probe_status\":\"" << decision.probe_status << "\"}\n";
}

}  // namespace musa_eval
