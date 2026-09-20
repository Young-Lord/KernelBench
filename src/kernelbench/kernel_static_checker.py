"""
Kernel Static Checker - Pattern-based validation for GPU kernel code.

The goal is flag reward hacking patterns (both strictly prohibited and possible ones).
through statically examining the code.

In the future we can add 
- AST-based detections 
- LM as a judge checker

Warning: This list is by no means complete and nor this is not a replacement for runtime checks.
We welcome feedback and contributions as community find new ways of hacks.

- Bypass hacks (PyTorch wrapping, try-except fallback, inheritance bypass)
- Disallow some high-level torch operations (depends on the settings)
- Backend implementation requirements, that CUDA or DSL features must be used

Usage:
    result = validate_kernel_static(code, backend="cuda")
    will return a tuple (valid, errors, warnings) 
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

def _strip_comments(code: str) -> str:
    """Remove # and // comments from code."""
    lines = []
    for line in code.split('\n'):
        # #include directives are C/C++ preprocessor statements, not comments
        if line.lstrip().startswith('#include'):
            lines.append(line)
            continue
        if '#' in line:
            line = line[:line.index('#')]
        if '//' in line:
            line = line[:line.index('//')]
        lines.append(line)
    return '\n'.join(lines)


# =============================================================================
# BYPASS CHECKS - Strictly Prohibited 
# some of this is from Kevin RL Paper (arxiv:2507.11948)
# =============================================================================

# --- Try-Except Fallback ---
# Rationale: Models wrap incomplete CUDA in exception handlers that fall back to PyTorch.
# This allows them to pass tests without actually implementing the kernel.
TRY_EXCEPT_PATTERNS = [r"\btry\s*:", r"\bexcept\s*:", r"\bexcept\s+\w+"]

# --- Pass Statement / Inheritance Bypass ---
# Rationale: Model inherits from reference class and uses 'pass' to do nothing,
# effectively just calling the parent implementation.
PASS_PATTERN = r"\bpass\b"

def check_code_bypass(code: str) -> Tuple[bool, str]:
    """
    Check for code bypass patterns (strictly prohibited).
    1. Try-Except Fallback: Models wrap incomplete CUDA in exception handlers
       that fall back to PyTorch when custom code fails.
    2. Pass Statement: Models inherit from reference and use 'pass' to do nothing,
       effectively calling parent implementation.
        Uses word boundary for 'pass' to avoid matching 'passed', 'bypass', etc.
    """
    code = _strip_comments(code)
    
    # Check for try-except fallback
    for pattern in TRY_EXCEPT_PATTERNS:
        if re.search(pattern, code):
            return (True, "Contains try-except block (potential fallback bypass)")
    
    # Check for pass statement
    if re.search(PASS_PATTERN, code):
        return (True, "Contains 'pass' statement (inheritance bypass)")
    
    return (False, "")

# Since KernelBench problems uses PyTorch as a reference, there could be settigs where
# Model generated code
# 1. Replaces some (not all) ops with custom kernels, others are kept in Torch
# --> More practical from a performance perspective (ie. make better systems) as you want to use whatever makes the best system for your use case. 
# 2. All compuational ops must be replaced with custom kernels
# --> Could be helpful from an eval (model ability on transpile + optimization) / RL training perspective 
# Depends the setting you use, you can move the checks below (pytorch_wrap, torch_computation_ops) 
# from WARNING to STRICT

# --- PyTorch NN Module Wrapping ---
# Allows: nn.Module, nn.Parameter, nn.ParameterList, nn.ParameterDict, 
#         nn.ModuleList, nn.ModuleDict, nn.init (needed for model structure)
# Blocks: nn.Linear, nn.Conv2d, nn.ReLU, etc. (compute layers)
PYTORCH_DISALLOWED_NN_PATTERN = r'torch\.nn\.(?!(Module|parameter|Parameter|ParameterList|ParameterDict|ModuleList|ModuleDict|init)\b)'

def check_pytorch_wrap(code: str) -> Tuple[bool, str]:
    """
    Check for PyTorch nn module usage (nn.Linear, nn.Conv2d, etc.).
    
    Allows containers (nn.Module, nn.Parameter, nn.init) needed for model structure.
    Blocks compute layers (nn.Linear, nn.Conv2d, nn.ReLU, etc.).
    """
    code = _strip_comments(code)
    if re.search(PYTORCH_DISALLOWED_NN_PATTERN, code):
        return (True, "Uses torch.nn compute layer (only containers, Parameter, init allowed)")
    return (False, "")


# --- Torch Computation Operations ---
# Rationale: These are high-level PyTorch ops that conduct computation.
# Using them directly defeats the purpose of writing custom kernels.
# Includes both torch.* and F.* (torch.nn.functional) patterns.
TORCH_COMPUTATION_OPS = [
    # Matrix operations
    "torch.mm", "torch.bmm", "torch.matmul", "torch.einsum",
    # Convolutions
    "torch.conv1d", "torch.conv2d", "torch.conv3d", "torch.conv",
    "torch.conv_transpose1d", "torch.conv_transpose2d", "torch.conv_transpose3d",
    # Pooling
    "torch.avg_pool1d", "torch.avg_pool2d", "torch.avg_pool3d",
    "torch.max_pool1d", "torch.max_pool2d", "torch.max_pool3d",
    "torch.adaptive_avg_pool1d", "torch.adaptive_avg_pool2d", "torch.adaptive_avg_pool3d",
    "torch.adaptive_max_pool1d", "torch.adaptive_max_pool2d", "torch.adaptive_max_pool3d",
    # Activations
    "torch.relu", "torch.hardtanh", "torch.elu", "torch.selu",
    "torch.leaky_relu", "torch.gelu", "torch.softsign", "torch.softplus",
    "torch.softmax", "torch.log_softmax", "torch.tanh", "torch.sigmoid",
    "torch.hardsigmoid", "torch.silu", "torch.mish",
    # Normalization
    "torch.batch_norm", "torch.group_norm", "torch.layer_norm",
    "torch.instance_norm", "torch.rms_norm", "torch.normalize",
    # Linear & Loss
    "torch.linear", "torch.cross_entropy", "torch.kl_div", "torch.mse_loss",
    "torch.huber_loss", "torch.triplet_margin_loss", "torch.cosine_similarity",
    # Others
    "torch.logsumexp", "torch.clamp", "torch.dropout",
]

# F.* patterns (torch.nn.functional equivalents)
TORCH_FUNCTIONAL_PATTERNS = [
    r"torch\.nn\.functional\.\w+",       # torch.nn.functional.*
    r"\bnn\.functional\.\w+",            # nn.functional.*
    r"\bF\.(conv|linear|relu|gelu|softmax|batch_norm|layer_norm|dropout|max_pool|avg_pool)",
]

def check_torch_computation_ops(code: str) -> Tuple[bool, str]:
    """
    Check for high-level torch computation operations.
    
    Matches both torch.* ops (torch.matmul) and F.* ops (F.relu).
    This check is optional/taste-based. Configure as needed.
    """
    code = _strip_comments(code)
    
    # Check torch.* ops
    torch_pattern = r'\b(' + '|'.join(re.escape(f) for f in TORCH_COMPUTATION_OPS) + r')(?=\s*\(|\s|$)'
    match = re.search(torch_pattern, code)
    if match:
        return (True, f"Uses torch computation op: {match.group(0)}")
    
    # Check F.* / nn.functional ops
    for pattern in TORCH_FUNCTIONAL_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return (True, f"Uses torch.nn.functional op: {match.group(0)}")
    
    return (False, "")

# --- Attention Entry Points ---
# Rationale: unlike `torch.matmul` or `nn.Linear`, one of these is never a
# weight container. A submission may legitimately hold a projection as an
# `nn.Linear` and compute with its weights, but calling a fused attention entry
# point *is* the core computation, so it is the one boundary a static check can
# draw without guessing where a task's kernel scope ends.
#
# These are A-tier checks only. The B tier is built around calling exactly these,
# which is why they are not in STRICT_CHECKS.
ATTENTION_ENTRY_POINTS = [
    r"\bscaled_dot_product_attention\s*\(",
    r"\bflash_attention_forward\s*\(",
    r"\befficient_attention_forward\s*\(",
    r"\bflash_attn\w*\s*\(",
    r"\bmemory_efficient_attention\s*\(",
]


def check_attention_entry_point(code: str) -> Tuple[bool, str]:
    """
    Check for a call to a fused attention entry point.

    Matched as a call rather than as a bare name: a submission is allowed to
    describe what it does not do, but not to do it.
    """
    code = _strip_comments(code)
    for pattern in ATTENTION_ENTRY_POINTS:
        match = re.search(pattern, code)
        if match:
            return (True, f"Calls a fused attention entry point: {match.group(0).rstrip('( ')}")
    return (False, "")

# =============================================================================
# Backend Specific Checks
# =============================================================================

# <========= CUDA CHECKS =========>
# Rationale: Valid CUDA kernels must have __global__ (kernel definition) and
# use load_inline or cpp_extension (PyTorch's inline compilation).
CUDA_COMPILE_PATTERNS = ["load_inline", "cpp_extension"]

def check_cuda_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid CUDA kernel implementation.
    
    Requirements:
    - Must have __global__ void kernel_name (kernel definition)
    - Must have load_inline or cpp_extension (PyTorch inline compilation)
    """
    code = _strip_comments(code)
    if "__global__" not in code:
        return (True, "Missing __global__ kernel definition")
    if not any(p in code for p in CUDA_COMPILE_PATTERNS):
        return (True, "Missing load_inline or cpp_extension for compilation")
    return (False, "")

# <========= HIP CHECKS =========>
# Rationale: Valid HIP kernels must have __global__ (kernel definition),
# use load_inline or cpp_extension (PyTorch's inline compilation) and
# use hipcc compiler
HIP_COMPILE_PATTERNS = ["load_inline", "cpp_extension"]

def check_hip_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid HIP kernel implementation.
    
    Requirements:
    - Must have __global__ void kernel_name (kernel definition)
    - Must have load_inline or cpp_extension (PyTorch inline compilation)
    """
    code = _strip_comments(code)
    if "__global__" not in code:
        return (True, "Missing __global__ kernel definition")
    if not any(p in code for p in HIP_COMPILE_PATTERNS):
        return (True, "Missing load_inline or cpp_extension for compilation")
    if "hipcc" not in code:
        return (True, "Missing hipcc compiler")
    return (False, "")

# <========= MUSA CHECKS =========>
MUSA_COMPILE_PATTERNS = ["load_inline", "cpp_extension", "MUSAExtension", "musa_extension"]

def check_musa_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid MUSA kernel implementation.

    Requirements:
    - Must have __global__ void kernel_name (kernel definition)
    - Must use load_inline, cpp_extension, MUSAExtension, or musa_extension for compilation
    - Must include either <musa_runtime.h> or <cuda_runtime.h> (torchada auto-translates the latter)

    Note: With torchada, CUDA sources (cuda_runtime.h) compile on MUSA without changes.
    """
    code = _strip_comments(code)
    if "__global__" not in code:
        return (True, "Missing __global__ kernel definition")
    if not any(p in code for p in MUSA_COMPILE_PATTERNS):
        return (True, "Missing load_inline, cpp_extension, or MUSAExtension for compilation")
    if "musa_runtime.h" not in code and "cuda_runtime.h" not in code:
        return (True, "Missing musa_runtime.h or cuda_runtime.h (torchada auto-translates the latter)")
    return (False, "")

# <========= TRITON CHECKS =========>
# Rationale: Triton kernels are compiled from @triton.jit decorated functions.
# They must use tl.* operations (tl.load, tl.store, etc.) for actual kernel work.
TRITON_JIT_PATTERN = r"@triton\.(jit|autotune)"
TRITON_OPS_PATTERN = r"\btl\.\w+"

def check_triton_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid Triton kernel implementation.
    
    Requirements:
    - Must have @triton.jit or @triton.autotune decorator
    - Must have tl.* operations (enforces actual Triton code, not wrapper)
    
    Note: Triton's compiler itself prevents PyTorch ops inside @triton.jit.
    """
    code = _strip_comments(code)
    if not re.search(TRITON_JIT_PATTERN, code):
        return (True, "Missing @triton.jit or @triton.autotune")
    if not re.search(TRITON_OPS_PATTERN, code):
        return (True, "No tl.* operations found in Triton kernel")
    return (False, "")


# <========= THUNDERKITTENS CHECKS =========>
# Rationale: ThunderKittens uses warp/warpgroup primitives and tile abstractions.
# Valid TK code must have namespace patterns and tile declarations.
TK_WARP_PATTERNS = [
    r"kittens::warp\b", r"kittens::warpgroup\b",
    r"::warpgroup::", r"::warp::", r"warpgroup::", r"warp::"
]
TK_TILE_PATTERN = r"(?:kittens::)?(?:st|rt)_\w+\s*<[^>]+>"

def check_tk_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid ThunderKittens kernel implementation.
    
    Requirements:
    - Must have warp/warpgroup namespace patterns (kittens::warp, etc.)
    - Must have tile declarations (st_bf<...>, rt_fl<...>, etc.)
    
    TODO: Add producer-consumer pattern check for complex kernels.
    """
    code = _strip_comments(code)
    if not any(re.search(p, code) for p in TK_WARP_PATTERNS):
        return (True, "Missing ThunderKittens warp/warpgroup patterns")
    if not re.search(TK_TILE_PATTERN, code):
        return (True, "Missing ThunderKittens tile declarations (st_*/rt_*)")
    return (False, "")


# <========= CUTE/CUTLASS CHECKS =========>
# CUTLASS uses cute:: namespace for tensor operations
# Check: https://github.com/NVIDIA/cutlass 
CUTE_PATTERNS = [
    r"cute::",           # cute:: namespace (CuTe library)
    r"cutlass::",        # cutlass:: namespace
    r"from cutlass",     # Python CUTLASS bindings
]

def check_cute_impl(code: str) -> Tuple[bool, str]:
    """Check for valid CUTLASS/CuTe kernel implementation."""
    code = _strip_comments(code)
    if not any(p in code for p in ["cute::", "cutlass::", "from cutlass"]):
        return (True, "Missing cute:: or cutlass:: namespace")
    return (False, "")


# <========= TILELANG CHECKS =========>
# TileLang uses TVM's T.prim_func decorator
# https://github.com/tile-ai/tilelang
TILELANG_PATTERNS = [
    r"@T\.prim_func",    # TVM primitive function decorator
    r"tvm\.build",       # TVM build call
    r"T\.grid",          # TileLang grid
]

def check_tilelang_impl(code: str) -> Tuple[bool, str]:
    """Check for valid TileLang kernel implementation."""
    code = _strip_comments(code)
    if not re.search(r"@T\.prim_func", code):
        return (True, "Missing @T.prim_func decorator")
    return (False, "")


# =============================================================================
# TIMING MANIPULATION CHECKS - Reward Hacking Patterns
# From adversarial hack PR and DeepReinforce blog
# =============================================================================

# <========= STREAM INJECTION CHECKS =========>
# Rationale: Models may use CUDA streams to defer computation, manipulate timing,
# or execute operations asynchronously to cheat benchmark measurements.
STREAM_PATTERNS = [
    r"torch\.cuda\.Stream\s*\(",       # torch.cuda.Stream()
    r"cuda\.Stream\s*\(",               # cuda.Stream() (if imported)
    r"with\s+torch\.cuda\.stream",      # with torch.cuda.stream(...)
    r"\.wait_stream\s*\(",              # .wait_stream() method
    r"\.record_stream\s*\(",            # .record_stream() method
]

def check_stream_injection(code: str) -> Tuple[bool, str]:
    """
    Check for CUDA stream injection patterns.
    
    Detects code that uses CUDA streams to potentially manipulate timing:
    1. Stream creation: torch.cuda.Stream(), cuda.Stream()
    2. Stream context managers: with torch.cuda.stream(...)
    3. Stream synchronization: .wait_stream(), .record_stream()
    
    Rationale: Streams can defer computation or change execution order,
    potentially affecting benchmark timing measurements.
    """
    code = _strip_comments(code)
    
    for pattern in STREAM_PATTERNS:
        if re.search(pattern, code):
            if "wait_stream" in pattern or "record_stream" in pattern:
                return (True, "Uses stream synchronization (potential timing manipulation)")
            elif "with" in pattern:
                return (True, "Uses stream context manager (potential timing manipulation)")
            else:
                return (True, "Uses CUDA streams (potential timing manipulation)")
    
    return (False, "")


# <========= THREAD INJECTION CHECKS =========>
# Rationale: Models may use threading to parallelize work or manipulate execution
# order in ways that could affect benchmark timing.
THREAD_PATTERNS = [
    r"threading\.Thread\s*\(",          # threading.Thread()
    r"import\s+threading",              # import threading
    r"from\s+threading\s+import",       # from threading import ...
    r"multiprocessing\.(Process|Pool|Manager|Queue|Pipe)",
    r"import\s+multiprocessing",        # import multiprocessing
    r"concurrent\.futures",             # concurrent.futures (thread pools)
    r"ThreadPoolExecutor",              # ThreadPoolExecutor
    r"ProcessPoolExecutor",             # ProcessPoolExecutor
]

def check_thread_injection(code: str) -> Tuple[bool, str]:
    """
    Check for thread/multiprocessing injection patterns.
    
    Detects code that uses threading or multiprocessing:
    1. Thread creation: threading.Thread()
    2. Threading imports: import threading
    3. Multiprocessing: Process, Pool, Manager, Queue, Pipe
    4. Concurrent futures: ThreadPoolExecutor, ProcessPoolExecutor
    
    Rationale: Threading can defer computation or change execution order,
    affecting timing measurements.
    
    Note: ProcessPoolExecutor might be legitimate in eval code but should
    not appear in kernel implementations.
    """
    code = _strip_comments(code)
    
    for pattern in THREAD_PATTERNS:
        if re.search(pattern, code):
            if "multiprocessing" in pattern:
                return (True, "Uses multiprocessing (potential timing manipulation)")
            elif "concurrent" in pattern or "Executor" in pattern:
                return (True, "Uses concurrent futures (potential timing manipulation)")
            else:
                return (True, "Uses threading (potential timing manipulation)")
    
    return (False, "")


# <========= LAZY EVALUATION CHECKS =========>
# Rationale: Models may create fake/lazy tensors that don't actually compute
# anything, passing correctness checks without real implementation.
LAZY_TENSOR_PATTERNS = [
    r"_make_subclass",                  # torch.Tensor._make_subclass (common lazy hack)
    r"class\s+\w+.*\(torch\.Tensor\)",  # Custom tensor subclasses
    r"class\s+\w+.*\(Tensor\)",         # Custom tensor subclasses (imported Tensor)
    r"torch\.Tensor\.__new__",          # Direct tensor construction (potential lazy)
]

def check_lazy_eval(code: str) -> Tuple[bool, str]:
    """
    Check for lazy tensor creation patterns.
    
    Detects patterns commonly used to create lazy/fake tensors:
    1. _make_subclass: Common way to create custom tensor subclasses
    2. Custom tensor subclasses: Classes inheriting from torch.Tensor
    3. Direct tensor construction: torch.Tensor.__new__ manipulation
    
    Rationale: Lazy tensors can pass correctness checks without actually
    computing anything, which is a form of reward hacking.
    """
    code = _strip_comments(code)
    
    for pattern in LAZY_TENSOR_PATTERNS:
        if re.search(pattern, code):
            if "_make_subclass" in pattern:
                return (True, "Uses _make_subclass (potential lazy tensor hack)")
            elif "class" in pattern:
                return (True, "Defines custom tensor subclass (potential lazy tensor hack)")
            else:
                return (True, "Uses direct tensor construction (potential lazy tensor hack)")
    
    return (False, "")


# <========= Timing Monkey Patch CHECKS =========>
# Rationale: Models may monkey-patch torch timing functions to fake benchmark results.
# This detects static patterns where timing functions are reassigned.
# especially when relying on timing markers like cuda.Event or time.perf_counter.
TIMING_EVENT_PATCH_PATTERNS = [
    r"torch\.cuda\.Event\.record\s*=",      # torch.cuda.Event.record = ...
    r"torch\.cuda\.Event\.elapsed_time\s*=", # torch.cuda.Event.elapsed_time = ...
    r"torch\.cuda\.synchronize\s*=",         # torch.cuda.synchronize = ...
    r"torch\.cuda\.Event\s*=",               # torch.cuda.Event = ... (replacing class)
    r"time\.perf_counter\s*=",               # time.perf_counter = ...
    r"time\.time\s*=",                       # time.time = ...
]

def check_timing_event_patch(code: str) -> Tuple[bool, str]:
    """
    Check for monkey patching of timing functions (static analysis).
    
    Detects code that reassigns torch or time functions used for benchmarking:
    1. torch.cuda.Event.record = ...
    2. torch.cuda.Event.elapsed_time = ...
    3. torch.cuda.synchronize = ...
    4. time.perf_counter = ...
    
    Rationale: Monkey patching timing functions can fake benchmark results,
    making slow code appear fast.
    """
    code = _strip_comments(code)
    
    for pattern in TIMING_EVENT_PATCH_PATTERNS:
        if re.search(pattern, code):
            return (True, "Reassigns timing function (monkey patch detected)")
    
    return (False, "")


# <========= PRECISION DOWNGRADE CHECKS =========>
# Rationale: Models may downgrade precision internally (FP32 -> FP16)
# to make kernels faster while still returning FP32 output. This is a reward
# hacking pattern where models cheat on timing while appearing correct.
# This might be a valid optimization, but is still worth flagging as a warning.
#
# We use explicit, high-confidence patterns that indicate intentional precision
# downgrading. These patterns have minimal false positives and clear semantic intent.

# Specific patterns that indicate FP32 -> FP16 precision downgrading
FP32_TO_FP16_PATTERNS = [
    # ========== CUDA / CUDA C++ ==========
    # 1.1 Explicit float -> half intrinsics (⭐ gold standard)
    # __float2half(f), __float2half_rn(f)
    r"__float2half(_rn)?\s*\(",
    
    # 1.2 Explicit C-style cast to __half
    # (__half)f
    r"\(\s*__half\s*\)\s*[\w\->\.]+",
    
    # 1.3 static_cast<half> / static_cast<__half>
    # static_cast<half>(f), static_cast<__half>(f)
    r"static_cast\s*<\s*(__half|half)\s*>\s*\(",
    
    # ========== Triton (Python) ==========
    # 2.1 Explicit tl.astype(..., tl.float16) (⭐ best signal)
    # tl.astype(x, tl.float16)
    r"tl\.astype\s*\(\s*[^,]+,\s*tl\.float16\s*\)",
    
    # ========== CUTLASS ==========
    # 3.1 NumericConverter float -> half (⭐ extremely reliable)
    # NumericConverter<half_t, float>
    r"NumericConverter\s*<\s*half_t\s*,\s*float\s*>",
    
    # 3.2 Epilogue output type = FP16
    # LinearCombination<half_t, ...>
    r"LinearCombination\s*<\s*half_t",
    
    # ========== CUTE (when used directly) ==========
    # 4.1 Explicit type conversion utility
    # type_convert<half_t>(x)
    r"type_convert\s*<\s*half_t\s*>\s*\(",
    
    # ========== ThunderKittens ==========
    # 5.1 Explicit half types in cast contexts
    # tk::half(x)
    r"tk::half\s*\(",
    
    # ========== Additional gaping patterns ==========
    # PyTorch explicit casts (common in inline CUDA contexts)
    r"\.half\s*\(",                              # .half() -> FP16
    r"\.float16\s*\(",                           # .float16() -> FP16
    r"\.to\s*\(\s*torch\.(float16|half)\b",      # .to(torch.float16)
    r"\.to\s*\(\s*dtype\s*=\s*torch\.(float16|half)\b",  # .to(dtype=torch.float16)
]

def check_precision_downgrade(code: str, precision: str = "fp32") -> Tuple[bool, str]:
    """
    Check for FP32 -> FP16 precision downgrading patterns in kernel code.
    
    Detects code that downgrades precision from FP32 to FP16 internally,
    which can make kernels faster but with reduced precision. This is a reward
    hacking pattern where models cheat on timing while appearing correct.
    
    Currently only checks for FP32 -> FP16 downgrades. Other precision combinations
    (e.g., FP16 -> BF16) are not flagged.
    
    Args:
        code: Kernel source code
        precision: Required precision - only checks when "fp32" or "float32"
    
    Returns:
        (True, error_message) if FP32 -> FP16 downgrade detected
        (False, "") if no downgrade detected
    
    Examples of detected patterns:
    - .half(), .float16()
    - .to(torch.float16), .to(torch.half)
    - dtype=torch.float16
    - __half, half2 (CUDA)
    - tl.float16 (Triton)
    """
    code = _strip_comments(code)
    precision = precision.lower()
    
    # Normalize precision to standard form
    precision_map = {"fp32": "fp32", "float32": "fp32", "fp16": "fp16", "bf16": "bf16", "bfloat16": "bf16"}
    precision = precision_map.get(precision, precision)
    
    # Only check for FP32 -> FP16 downgrades
    if precision != "fp32":
        return (False, "")
    
    # Check for FP16 patterns
    for pattern in FP32_TO_FP16_PATTERNS:
        if re.search(pattern, code):
            return (True, "Precision downgrade detected: required FP32 but code uses FP16")
    
    return (False, "")

# =============================================================================
# In the future, we can add a AST-based checker and a LM-as-a-judge checker
# =============================================================================


# =============================================================================
# B-TIER (LIBRARY DISPATCH) CHECKS
#
# The checks above assume the submission must implement everything in a device
# kernel, so calling a torch compute op is a hack. The B tier inverts that
# premise: calling a whitelisted library *is* the task, and the hacks become
# calling something outside the whitelist, picking a path from a version string
# instead of a runtime probe, and never emitting a dispatch trace.
#
# These are configured from the task contract's `library_policy` block and are
# wired up by `validate_library_kernel_static` below.
# =============================================================================

# Modules that are plumbing rather than a compute library: tensor allocation,
# dtype casts, module structure, and the standard library. Matched on the
# top-level module name, so every `torch.*` submodule counts as plumbing and the
# actual compute-op question stays with `check_torch_computation_ops`.
LIBRARY_PLUMBING_ROOTS = {
    "abc", "argparse", "collections", "contextlib", "copy", "ctypes",
    "dataclasses", "enum", "functools", "importlib", "itertools", "json",
    "logging", "math", "operator", "os", "pathlib", "random", "re", "string",
    "sys", "textwrap", "time", "typing", "warnings",
    "numpy", "np",
    "torch",
}

IMPORT_PATTERNS = [
    r"^\s*import\s+([A-Za-z_][\w.]*)",
    r"^\s*from\s+([A-Za-z_][\w.]*)\s+import",
    r"""importlib\.import_module\s*\(\s*["']([A-Za-z_][\w.]*)["']""",
]

# Rationale: pulling in a compiled library by path sidesteps the import scan.
CTYPES_LOAD_PATTERN = r"""CDLL\s*\(\s*["']([^"']+)["']"""

# Rationale: the task contract forbids choosing a dispatch path from a driver,
# Toolkit or library version string. Capability must be probed at runtime.
# `parse_version` / `LooseVersion` are matched as bare names on purpose: merely
# importing one signals the intent to compare versions.
VERSION_DISPATCH_PATTERNS = [
    r"\b__version__",
    r"\btorch\.version\b",
    r"\bparse_version\b",
    r"\bget_version\s*\(",
    r"\bversion\s*(?:==|!=|>=|<=|>|<)\s*[\"']",
    r"\bLooseVersion\b|\bStrictVersion\b",
]


def _allowed_library_roots(allowed_libraries: Optional[List[str]]) -> set:
    """Normalize `allowed_libraries` to import-root names.

    `libmudnn` is accepted as the import name `mudnn`, so a task contract can
    name either the linker library or the Python module.
    """
    roots = set()
    for library in allowed_libraries or []:
        normalized = library[3:] if library.lower().startswith("lib") else library
        roots.add(normalized.lower())
    return roots


def check_library_whitelist(
    code: str,
    allowed_libraries: Optional[List[str]] = None,
    allowed_symbol_prefixes: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Check that every library the submission pulls in is whitelisted.

    A module is accepted when its top-level name is plumbing, is exactly an
    entry of `allowed_libraries` (with a leading `lib` stripped), or contains one
    of `allowed_symbol_prefixes`. The substring rule is what lets a contract
    whitelist `["musa", "mudnn"]` and thereby accept `torch_musa` and `mudnn`.
    """
    source = _strip_comments(code)
    allowed_roots = _allowed_library_roots(allowed_libraries)
    prefixes = [prefix.lower() for prefix in allowed_symbol_prefixes or []]

    def is_allowed(module_name: str) -> bool:
        root = module_name.split(".")[0].lower()
        if root in LIBRARY_PLUMBING_ROOTS or root in allowed_roots:
            return True
        return any(prefix in root for prefix in prefixes)

    for pattern in IMPORT_PATTERNS:
        for match in re.finditer(pattern, source, flags=re.MULTILINE):
            module_name = match.group(1)
            if not is_allowed(module_name):
                return (True, f"Imports non-whitelisted library: {module_name}")

    for match in re.finditer(CTYPES_LOAD_PATTERN, source):
        library_path = match.group(1)
        stem = re.split(r"[/\\]", library_path)[-1].split(".")[0]
        if not is_allowed(stem):
            return (True, f"Loads non-whitelisted shared library: {library_path}")

    return (False, "")


def check_version_string_dispatch(code: str) -> Tuple[bool, str]:
    """Check for dispatch decisions keyed on a version string."""
    source = _strip_comments(code)
    for pattern in VERSION_DISPATCH_PATTERNS:
        match = re.search(pattern, source)
        if match:
            return (True, f"Dispatches on a version string: {match.group(0)}")
    return (False, "")


def check_dispatch_trace_emission(
    code: str,
    required_trace_fields: Optional[List[str]] = None,
    trace_env_var: Optional[str] = None,
    case_id_env_var: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check that the submission can emit a dispatch trace.

    The trace is what makes the B tier gradeable, so a submission that never
    mentions the required field names cannot produce one. Requiring the
    environment variables as well matters more than it looks: field names can
    appear in a docstring, an env var lookup cannot — it is the thing that
    actually locates the file the evaluator will read, and the case identity the
    record has to carry.

    The case-id variable is checked because a trace record must name the case it
    belongs to and the evaluator compares that against the case it asked for. A
    submission that never reads it can only be guessing.

    This is a structural proxy, not a proof: the runtime trace comparison is
    what actually decides.

    Args:
        code: submission source
        required_trace_fields: field names the trace records must carry
        trace_env_var: the environment variable locating the trace file
        case_id_env_var: the environment variable naming the current case
    """
    source = _strip_comments(code)
    missing = [
        field for field in required_trace_fields or []
        if f'"{field}"' not in source and f"'{field}'" not in source
    ]
    if missing:
        return (True, "dispatch trace is missing required fields: " + ", ".join(missing))

    for env_var, purpose in ((trace_env_var, "locates the trace file"), (case_id_env_var, "names the current case")):
        if env_var and f'"{env_var}"' not in source and f"'{env_var}'" not in source:
            return (True, f"dispatch trace never reads {env_var}, which {purpose}")

    return (False, "")


# =============================================================================
# REGISTRY & PRESETS
# =============================================================================

# Check functions can take either (code) or (code, precision) arguments
# Most checks take only code, but precision-dependent checks take both
CHECK_FUNCTIONS: Dict[str, Union[Callable[[str], Tuple[bool, str]], Callable[[str, str], Tuple[bool, str]]]] = {
    # Bypass checks (strict)
    "code_bypass": check_code_bypass,
    "pytorch_wrap": check_pytorch_wrap,
    "timing_event_patch": check_timing_event_patch,  # clearly malicious
    
    # Torch ops (depends on your setups)
    "torch_computation_ops": check_torch_computation_ops,
    
    # Timing manipulation checks (usually warnings)
    "stream_injection": check_stream_injection,
    "thread_injection": check_thread_injection,
    "lazy_eval": check_lazy_eval,
    "precision_downgrade": check_precision_downgrade,  # precision-dependent
    
    # Backend-specific implementation checks
    # should be strict
    "cuda_impl": check_cuda_impl,
    "hip_impl": check_hip_impl,
    "musa_impl": check_musa_impl,
    "triton_impl": check_triton_impl,
    "tk_impl": check_tk_impl,
    "cute_impl": check_cute_impl,
    "tilelang_impl": check_tilelang_impl,

    # A-tier checks. Not in STRICT_CHECKS because the B tier is built around
    # calling exactly what this one forbids.
    "attention_entry_point": check_attention_entry_point,
}

# What the A tier adds on top of the strict checks.
#
# The A tier forbids library compute, but "the core computation" has no single
# definition: an answer may hold a projection as an `nn.Linear` and compute with
# its weights, which is a container, or call it, which is compute, and no static
# check can tell those apart. The attention entry point can be told apart, so it
# is the boundary this tier enforces; every task's contract states the rest of
# its kernel scope under `kernel_scope` for a reader rather than for this check.
A_TIER_FORBIDDEN_CHECKS = [
    "attention_entry_point",
]

# Checks that require additional parameters beyond just code
PRECISION_DEPENDENT_CHECKS = {"precision_downgrade"}

# Here are some presets for you to use
# You are welcome to adapt them to your settings
# These checks are NECESSARY for all kernels (strict = error)
STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",  
    "lazy_eval",         
]

# Backend-specific checks are added later at entry point
# per backend implementation check, usually strict
BACKEND_IMPL_CHECK = {
    "cuda": "cuda_impl",
    "hip": "hip_impl",
    "musa": "musa_impl",
    "triton": "triton_impl",
    "thunderkittens": "tk_impl",
    "cute": "cute_impl",
    "cutlass": "cute_impl",  # alias
    "tilelang": "tilelang_impl",
}

# These are optional checks (by user's decision) - flagged as warnings
# Move to STRICT_CHECKS if you want to enforce them
WARNING_CHECKS: List[str] = [
    # up to user to allow program to still have some torch computation ops
    "pytorch_wrap",
    "torch_computation_ops",  
    "stream_injection",       # could have legitimate uses (async ops), but should be careful!
    "precision_downgrade",    # precision downgrading - can be intentional but often a hack
]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def validate_kernel_static(
    code: str,
    backend: str = "cuda",
    precision: str = "fp16",
    forbidden: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate kernel code through statically inspecting the code
    We configure the checks against check groups that we have provided for common hacks.
    Note we do not guarantee that all checks are exhaustive. This is also only on the static level.
    
    Args:
        code: Kernel source code
        backend: "cuda", "hip", "triton", or "thunderkittens"
        precision: "fp16", "fp32", or "bf16" (for future precision checks)
        forbidden: Check categories that cause errors (default: STRICT_CHECKS)
        warnings: Check categories that cause warnings (default: WARNING_CHECKS)
        
    Returns:
        (valid, errors, warnings)
        valid: bool
        errors: List[str]
        warnings: List[str]
    """
    # Copy defaults to avoid mutating global lists
    forbidden_checks = list(forbidden) if forbidden is not None else list(STRICT_CHECKS)
    warning_checks = list(warnings) if warnings is not None else list(WARNING_CHECKS)
    
    # Add backend implementation check if specified
    if backend in BACKEND_IMPL_CHECK:
        impl_check = BACKEND_IMPL_CHECK[backend]
        if impl_check not in forbidden_checks:
            forbidden_checks.append(impl_check)
    
    # Aggregate results
    errors: List[str] = []
    warnings_list: List[str] = []
    
    for check_name in set(forbidden_checks + warning_checks):
        if check_name not in CHECK_FUNCTIONS:
            continue
        
        # Handle precision-dependent checks
        if check_name in PRECISION_DEPENDENT_CHECKS:
            has_issue, msg = CHECK_FUNCTIONS[check_name](code, precision)
        else:
            has_issue, msg = CHECK_FUNCTIONS[check_name](code)
        
        if has_issue:
            if check_name in forbidden_checks:
                errors.append(msg)
            else:
                warnings_list.append(msg)
    
    valid = len(errors) == 0 # valid if no errors
    return valid, errors, warnings_list


def validate_library_kernel_static(
    code: str,
    policy: dict,
    backend: str = "musa",
    precision: str = "fp16",
    forbidden: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """Validate a B-tier (library dispatch) submission against a task contract.

    Reuses the A-tier strict checks and swaps the premise, because the B tier is
    graded on dispatching to a whitelisted library rather than on implementing
    everything in a device kernel:

    - calling library compute is the task, so `torch_computation_ops` and
      `pytorch_wrap` stay warnings instead of becoming errors;
    - the backend implementation check only runs when the submission actually
      defines a device kernel, since the fused and composition paths need none;
    - three B-tier rules become errors: every imported library must be
      whitelisted, dispatch must not key on a version string, and the required
      dispatch-trace fields must be present.

    Args:
        code: submission source
        policy: the task contract's `library_policy` block. Reads
            `allowed_libraries`, `allowed_symbol_prefixes` and
            `required_trace_fields`.
        backend: backend name for the optional implementation check
        precision: forwarded to the precision-dependent checks
        forbidden: override the strict check set
        warnings: override the warning check set

    Returns:
        (valid, errors, warnings)
    """
    allowed_libraries = policy.get("allowed_libraries", [])
    allowed_symbol_prefixes = policy.get("allowed_symbol_prefixes", [])
    required_trace_fields = policy.get("required_trace_fields", [])
    trace_env_var = policy.get("trace_env_var")
    case_id_env_var = policy.get("case_id_env_var")

    # An empty backend skips the backend implementation check in
    # validate_kernel_static; B-tier code is not required to define a kernel.
    valid, errors, warnings_list = validate_kernel_static(
        code,
        backend="",
        precision=precision,
        forbidden=forbidden,
        warnings=warnings,
    )

    for has_issue, message in (
        check_library_whitelist(code, allowed_libraries, allowed_symbol_prefixes),
        check_version_string_dispatch(code),
        check_dispatch_trace_emission(code, required_trace_fields, trace_env_var, case_id_env_var),
    ):
        if has_issue:
            errors.append(message)

    if "__global__" in _strip_comments(code):
        impl_check_name = BACKEND_IMPL_CHECK.get(backend)
        if impl_check_name and impl_check_name in CHECK_FUNCTIONS:
            has_issue, message = CHECK_FUNCTIONS[impl_check_name](code)
            if has_issue:
                errors.append(message)

    return len(errors) == 0, errors, warnings_list


# =============================================================================
# TIER DISPATCH
#
# A task declares its tier next to its reference model: a problem file may define
# module-level `TIER` and `LIBRARY_POLICY`. KernelBench executes the problem
# source into a namespace, so both end up in that namespace and the evaluator can
# read them from there. Absent metadata means the A tier, which is what every
# pre-existing problem is.
# =============================================================================

KNOWN_TIERS = ("A_kernel", "B_library")


def resolve_tier_and_library_policy(namespace: dict) -> Tuple[str, Optional[dict]]:
    """Read the tier and library policy out of an executed problem namespace.

    Args:
        namespace: the namespace the problem source was executed into

    Returns:
        (tier, library_policy), with `library_policy` None for the A tier.

    Raises:
        ValueError: unknown tier, or a B-tier problem that declares no policy.
            A B-tier task without a whitelist cannot be graded, so failing loudly
            beats silently grading it as A.
    """
    tier = namespace.get("TIER", "A_kernel")
    if tier not in KNOWN_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {KNOWN_TIERS}")

    library_policy = namespace.get("LIBRARY_POLICY")
    if tier == "B_library" and not isinstance(library_policy, dict):
        raise ValueError("a B_library problem must define a LIBRARY_POLICY dict")
    return tier, library_policy


def static_audit_kernel(
    code: str,
    tier: str = "A_kernel",
    library_policy: Optional[dict] = None,
    backend: str = "cuda",
    precision: str = "fp16",
    forbidden: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """Run the static audit appropriate to the task's tier.

    Single entry point so callers do not have to remember that the two tiers
    check opposite things: the A tier polices library use, the B tier requires it.

    Args:
        code: submission source
        tier: "A_kernel" or "B_library"
        library_policy: required for the B tier
        backend: backend name, used by the A-tier implementation check
        precision: "fp16", "fp32" or "bf16", for the precision-dependent checks
        forbidden / warnings: forwarded check-set overrides

    Returns:
        (valid, errors, warnings)
    """
    if tier == "B_library":
        if not isinstance(library_policy, dict):
            raise ValueError("the B_library tier requires a library_policy")
        return validate_library_kernel_static(
            code,
            library_policy,
            backend=backend,
            precision=precision,
            forbidden=forbidden,
            warnings=warnings,
        )

    if tier != "A_kernel":
        raise ValueError(f"unknown tier {tier!r}; expected one of {KNOWN_TIERS}")

    # The A tier adds its own checks to the shipped defaults. `STRICT_CHECKS`
    # does not police library compute on its own, which is deliberate upstream:
    # a stock KernelBench task is graded on speedup, and its author may be
    # content to let a submission wrap a torch op as long as the result is
    # faster. This tier is not, so the attention entry point is an error here.
    if forbidden is None:
        forbidden = list(STRICT_CHECKS) + A_TIER_FORBIDDEN_CHECKS

    return validate_kernel_static(
        code,
        backend=backend,
        precision=precision,
        forbidden=forbidden,
        warnings=warnings,
    )
