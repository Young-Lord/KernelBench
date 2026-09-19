#include "runner_api.h"

#include <stdexcept>

namespace musa_eval {

std::vector<Tensor> run_sdpa(const std::vector<Tensor>& inputs,
                             const RunAttributes& attributes,
                             const DispatchDecision& decision) {
  (void)inputs;
  (void)attributes;
  (void)decision;
  // BEGIN_AGENT_EDIT:HOST_LAUNCH
  // Route to an allowed fused-library adapter, allowed library composition,
  // or launch the custom fallback kernel. Return output and optional LSE.
  throw std::runtime_error("starter launch path is not implemented");
  // END_AGENT_EDIT:HOST_LAUNCH
}

}  // namespace musa_eval
