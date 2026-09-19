#include <musa_runtime.h>

// BEGIN_AGENT_EDIT:FALLBACK_KERNEL
// Implement only the configurations rejected by the allowed library paths.
// The evaluator rejects renamed third-party kernels and host-side tensor math.
extern "C" __global__ void sdpa_fallback_placeholder() {}
// END_AGENT_EDIT:FALLBACK_KERNEL
