# Frozen tensor ABI

The runner receives an input directory and an output directory. Each directory
contains `tensors.json` plus one raw little-endian file per tensor.

Each tensor entry contains `name`, `file`, logical `dtype`, `shape`, `layout`,
`byte_order`, `nbytes`, and SHA-256. Files are dense C-order arrays with no
header or padding. `bfloat16` is stored as IEEE bfloat16 bit patterns in
little-endian unsigned 16-bit words.

The input manifest contains `q`, `k`, and `v`. The output manifest must contain
`output` and may contain FP32 `lse`. B-tier runners also write one JSON Lines
dispatch record per case to the evaluator-provided trace path.

The native runner owns all host and device allocation and uses the current MUSA
stream. It must not link libtorch or ATen.
