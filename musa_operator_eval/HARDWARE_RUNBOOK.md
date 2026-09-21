# MUSA task policy

The rules that code does not enforce. How to probe a device, take a measurement
or drive a case is described by the tool that does it -- repeating any of that
here would only create a copy that drifts away from the source.

## Measurements belong to an environment
Every measurement is tied to a snapshot from `tools/collect_environment.py`. The
snapshot carries two identifiers because two different things can change:

- `snapshot_id` identifies the configuration: device model, architecture,
  driver, toolkit components, torch stack. Two machines built the same way share
  it.
- `device_instance_id` identifies the physical card, derived from its GPU UUID.
  On a rented host this is the identifier that changes when an instance comes up
  elsewhere, which is the failure a container image digest cannot detect at all.

Latencies may only be compared between runs whose `snapshot_id` matches, and
reused across `device_instance_id` values only when the instance is known to be
pinned to a single host.

A container image digest is recorded when the platform exposes one and set to
`null` with an `image_digest_unavailable_reason` otherwise. A missing digest is
not an error -- it would not have covered the host-side driver, the GPU identity
or the cgroup limits, all of which are collected directly.

## How a number is measured

`measure_baseline.py` does not carry its own timing loop: it calls
`kernelbench.timing.time_execution_with_cuda_event` and `get_timing_stats`, the same
two functions a `run_task.py` report is produced by. That matters because a baseline's
`latency_ms` is a denominator and a report's `runtime` is a numerator -- two protocols
would make every ratio a ratio between protocols rather than between implementations.

What the framework's loop does, and therefore what every recorded number means:

* warm up the callable, empty the caching allocator, then run **one pass** of
  `num_trials` (100) trials;
* **thrash L2 before every trial** (a 256 MB fill), so the numbers are **cold-cache**
  numbers;
* discard the first trial (`discard_first=1`);
* time each trial with device events, not a host clock;
* report the **mean** of the trials, which is what `KernelExecResult.runtime` is.

`get_timing_stats` renders its summary as `f"{value:.3g}"` -- three **significant**
digits, not three decimals -- so a recorded `latency_ms` of `1.27` is the mean
`1.2719768`, and `0.129` is `0.1288391`. Comparing a recorded number against a sample
list means reproducing that format rather than rounding to three decimals.

A record says this in its `measurement_protocol` block and names the function that
defines it (`source`), carries the trials it used (`latency_ms_samples`) and the
framework's summary of them (`latency_ms_stats`). A case manifest no longer declares a
protocol of its own: the `performance` block it used to carry was read by nothing and
had drifted to a different protocol, which is the same failure mode as the per-case
tolerance that only one grader read.

Two consequences worth stating plainly:

* A number measured by a *different* protocol is not comparable with these, and the
  case manifests no longer offer one to be measured by. The compiled path's runner now
  takes the same measurement (event pair, per-call L2 flush, first trial discarded,
  mean) and says so in the `timing` block of every case row, so a compiled number is
  comparable with a baseline; what still separates the two forms is not the clock but
  the ABI, whose host-to-device copies sit inside the measured interval, which is why a
  compiled case is only ranked against another compiled case.
* Upstream KernelBench's numbers are cold-cache and mean-based, like ours, so they are
  the same kind of number; but a number from a warm-cache tool (this repository's own
  earlier records, for instance) is not.

## The hidden case set
A submission is graded on the hidden set; the public one exists so something can
be run before submitting. `run_task.py` evaluates whatever `--cases` points at,
defaulting to the task's `public_cases.json`:

```bash
# The hidden goldens are generated once, outside the agent-visible tree, and the
# runner resolves each case's declared `golden` path against --generated-dir.
python musa_operator_eval/tools/generate_cases.py \
  --task-dir musa_operator_eval/tasks/sdpa_forward_pilot \
  --manifest musa_operator_eval/private/sdpa_forward_b_v0/cases.private.json \
  --output-dir musa_operator_eval/private/generated/sdpa_forward_b_v0

python musa_operator_eval/evaluator/run_task.py \
  --task-dir musa_operator_eval/tasks/sdpa_forward_pilot \
  --cases musa_operator_eval/private/sdpa_forward_b_v0/cases.private.json \
  --generated-dir musa_operator_eval/private/generated/sdpa_forward_b_v0 \
  --submission <model_new.py> --backend musa --precision fp16 \
  --output runs/sdpa_forward_b_v0/hidden_report.json
```

The run reports its stages in order and stops at the first failure: case
generation, golden, static audit, build, link whitelist check (B tier only),
evaluation. `--build-dir` is where the submission's compiled artifacts land, which
is what the link check reads; it defaults to `<generated-dir>/build`.

The hidden goldens live under `musa_operator_eval/private/generated/<task id>/` on the
device, and `--generated-dir` points at that directory. They are regenerated rather
than committed, and a sync that deletes files the local tree does not have will
delete them: exclude `musa_operator_eval/private/generated/` when syncing.

§4.7 re-runs every correctness and boundary case with the same seed and reports a case
that disagrees with itself as a stability failure; `--stability-reruns 0` turns that
off. Every full report also carries an `environment` block comparing the machine
against the snapshot the task names, and a run on a different configuration stops
before the build.

The hidden manifest is built by `make_private_manifest.py` from recorded gap
evidence. It refuses to build a manifest that does not cover at least
`minimum_gap_cases` gaps across at least `required_gap_reasons` categories, both
read from the task contract.

## The compiled path

Every package ships `starter/cpp`: a frozen `runner.cc`, `abi_io.h` and `build.sh`
plus one editable `kernel.mu`. The runner reads `<input-dir>/tensors.json` and its
`.bin` files, calls `kernel_entry`, and writes `<output-dir>/tensors.json` and its
`.bin` files, without libtorch -- the artifact's dynamic section names `libmusart`
and the C runtime.

    # A: the worked submission that proves the ABI (it lives under private/ because
    # it is an answer, not a scaffold)
    python musa_operator_eval/evaluator/run_binary.py \
        --task-dir musa_operator_eval/tasks/kb_l1_97_a \
        --submission musa_operator_eval/private/scaled_dot_product_attention_a_v0/compiled \
        --cases musa_operator_eval/private/scaled_dot_product_attention_a_v0/cases.private.json \
        --generated-dir musa_operator_eval/private/generated/scaled_dot_product_attention_a_v0 \
        --output /tmp/binary_a.json

    # B: the same driver, where the link check finally has a real artifact to read
    python musa_operator_eval/evaluator/run_binary.py \
        --task-dir musa_operator_eval/tasks/kb_l3_43_b \
        --submission musa_operator_eval/tasks/kb_l3_43_b/starter/cpp \
        --cases musa_operator_eval/private/mingpt_causal_attention_b_v0/cases.private.json \
        --generated-dir musa_operator_eval/private/generated/mingpt_causal_attention_b_v0

Two things to know about it. The stage list is §4.7's, and `link_whitelist_check`
appears only for the B tier. The build script gets `--extra-library` for whatever
the contract whitelists and the machine can resolve, so what is linked and what is
checked are the same list.

The compiled path grades what a case's input directory holds and nothing else, which is
the one thing to know before writing an adapter. A reference that keeps weights in
`__init__` cannot be rebuilt by a torch-free submission -- they come from the case's
seed and the construction order -- so for those eight packages the evaluator
materializes the state with the framework's own construction, stages it beside the
case's tensors as `state_*`, and records a digest per case row:

    "model_state": {"materialized": true, "digest": "4f861e01...", "tensors": 5,
                    "dtype": "float16", "seed": 91001, "tensor_prefix": "state_"}

Names are the module's own keys behind the prefix, so a submission reads
`state_c_attn.weight`, `state_c_attn.bias`, `state_c_proj.weight`, `state_c_proj.bias`
and the causal `state_bias` buffer of a MiniGPT attention layer. The digest covers the
whole state: two runs of one case must agree on it, and a disagreement means the
materialization drifted (a different torch, a changed construction) rather than that a
kernel is wrong. A case whose reference has no state stages nothing and reports
`materialized: false`, and the runner is handed the case's own input directory.

`tests/test_materialize_model_state.py` holds both halves: the driver's staging against
a stand-in materializer (no torch needed), and, on a machine with the framework, the
tool's state against the state the framework's own loader builds from the same seed.

### What `runtime` measures here, and why it needed a flag

A compiled case runs in a process that starts cold. Measured on the target device: the
first call in a fresh process pays **237 ms** for the device context plus roughly as much
again for mapping a 2.2 GB muDNN; reading and writing the case's files costs **58-470 ms**
depending on size (about 110 MB/s); the attention call itself is **0.5 ms** the first
time and **under 0.1 ms** after that. A wall clock around the process is therefore 99.9%
startup and disk, and it is not the quantity the Python path's `runtime` reports.

The runner takes `--warmup W --repeat N` (defaults 3 and 100, the framework's own call
counts): it reads the inputs once, calls `kernel_entry` W times to settle, then takes
`repeat + 1` timed calls -- a MUSA event pair around each one, the L2 cache flushed
before each one, the first discarded -- and prints one line of per-call times. The
driver takes the **mean** the framework's `get_timing_stats` takes and records it as
`runtime`, keeping the process's wall clock as `wall_ms`.

    "runtime": 1.28,                     # mean of the timed calls, milliseconds
    "wall_ms": 553.1,                    # the whole process: start, load, read, write
    "timing": {"warmup": 3, "repeat": 100, "discarded": 1, "timer": "musa_event",
               "cache": "cold", "l2_thrash_bytes": 268435456,
               "mean_ms": 1.28, "min_ms": 1.19, "samples": [1.28, 1.31, ...]}

The protocol travels with the number: `timer`, `cache` and `l2_thrash_bytes` say how it
was taken, so a row cannot claim a cold-cache event measurement it did not make (an
older runner, or a device where the 256 MB flush buffer could not be allocated, records
`cache: "warm"`). One boundary is worth knowing here: the events are recorded on the
default stream, so a submission that launches on a stream of its own and does not
synchronize is timed as if it had done nothing. Every worked answer and skeleton uses
the default stream, which is the pattern the ABI documents.

Two rules follow from that shape. A submission that re-initialises the device inside
every call is measured paying for it -- hoisting it into a function-local static is the
submission's own business, and the warmup exists to make that possible. And a compiled
case is only ever ranked against another compiled case: the two forms have different
overheads, which is now a visible fact rather than a hidden one.

`private/scaled_dot_product_attention_b_v0/compiled` is the B tier's worked
submission -- the one that shows the compiled path can score, not just build:

    # B, graded: 10 hidden cases, 10 passed, every recorded path its expected one
    python musa_operator_eval/evaluator/run_binary.py \
        --task-dir musa_operator_eval/tasks/kb_l1_97_b \
        --submission musa_operator_eval/private/scaled_dot_product_attention_b_v0/compiled \
        --cases musa_operator_eval/private/scaled_dot_product_attention_b_v0/cases.private.json \
        --generated-dir musa_operator_eval/private/generated/scaled_dot_product_attention_b_v0 \
        --precision fp16 --output /tmp/binary_b.json

## Calling muDNN from a submission

The library is on the machine and needs no framework. What it takes to use it:

* **Headers.** `/usr/local/musa-3.1.0/include/mudnn*.h`. `mudnn_nn.h` carries
  `musa::dnn::ScaledDotProductAttention` (`SetHeadsNum`, `SetEmbedDim`, `SetCausal`,
  `SetMaskMode`, `SetKeyFormat`, `RunFlash` for the fused kernel, `RunMath` for the
  non-fused one) and `musa::dnn::MultiHeadAttention`; `mudnn_math.h` and
  `mudnn_base.h` carry the GEMM, softmax and tensor/handle types. `libmudnn.so`
  exports 10001 `musa::dnn::*` symbols.
* **Link.** Only `libmudnn` and `libmusart` are needed. A `g++ -std=c++17` binary
  linking those two and nothing else gets `SUCCESS` from `RunFlash`, which is what
  makes a §4.6-compliant B submission possible at all.
* **`SetEmbedDim` is the model width, not the head dimension** -- `num_heads *
  head_dim`. Passing `head_dim` comes back as `INVALID_PARAMETER ... Validate q
  dim`, which reads like a shape complaint and is not one.
* **The fused boundary is the library's to draw.** On this build `RunFlash` answers
  head dimension up to 160 and refuses above it: *"Flash Attention 2 Not Support
  HeadDim > 160 Now"*. Ask it and read the status rather than predicting from a
  shape or a version string.
* **Descriptors carry the layout.** `Tensor::SetNdInfo(ndims, dims, strides)` takes
  `int64_t` arrays -- the `int` overloads do not exist, and passing `int` fails to
  compile -- and strides are how a `[B, H, S, D]` view is expressed without moving
  data.
* **The region rule reaches includes.** The compiled-source audit flags any line
  naming a library (`mudnn` among them) outside `probe_and_dispatch` and
  `host_launch`, and the flag is on the *line*, so `#include <mudnn.h>` has to sit
  inside one of those regions in a single-file submission.

A minimal standalone probe -- outside the harness, no torch, a few dozen lines -- is
the fastest way to learn a descriptor's rules; `RunFlash` returning `SUCCESS` on a
tiny case is what settled the `SetEmbedDim` question above.

## Four traps worth knowing before reading a number

**The graded bar is resolved from the contract, not from a manifest.** A task's
`tolerances.max_abs_error` is a loosen-only override of the framework's per-dtype floor
(`get_tolerance_for_precision`: 1e-4 at fp32, 1e-2 at fp16/bf16), applied to `atol` and
`rtol` alike. Both drivers resolve it the same way and every report records the value as
`tolerance_used`, so a reader never has to guess which bar a row was graded at. A
per-case `tolerance` field existed once and only the compiled driver read it, which is
exactly how two graders end up holding one submission to two different numbers.

**`runtime` is milliseconds, whatever the framework's comment says.**
`kernelbench.eval`'s `KernelExecResult` documents `runtime` as microseconds, and the
values it actually records are milliseconds: an expert module measures 0.834 in a
`run_task.py` report and 0.7048 in the baseline record for the same case, and those
are the same number. `rank.py` therefore divides the two directly. A tool that
trusted the comment reported every ratio a thousand times too high.

**A killed run can leave torchada's JIT lock behind, and every later run then hangs
before printing anything.** `torchada` compiles its C++ ops on import through torch's
`cpp_extension`, which takes a file baton at
`~/.cache/torch_extensions/py310_cpu/torchada_cpp_ops/lock`. A `kill -9` during that
compile leaves the file with no owner and no holder to release it, and a fresh
`import kernelbench.gpu` then blocks in `FileBaton.wait` indefinitely -- with no
output and no error, which reads exactly like a device hang. Clear it:

    rm -f ~/.cache/torch_extensions/py310_cpu/torchada_cpp_ops/lock \
          ~/.cache/torch_extensions/py310_cpu/torchada_cpp_ops/.ninja_lock

and prefer `kill` to `kill -9`, or check `pgrep -af "evaluator/run_task[.]py"` and
`mthreads-gmi` before concluding anything about the machine.

**A truncated search is not evidence of absence.** `find / -iname 'mudnn*' | head
-10` returns `torch_musa` paths first, and the SDK's own headers only appear after
them -- so a search that stops early reads as "the header does not exist". That
conclusion was drawn once here and it was wrong: muDNN's headers have been in
`/usr/local/musa-3.1.0/include/` the whole time, and the claim that no host entry
point existed delayed the B tier's compiled path by weeks. Scope the search (`find
/ -xdev ...`), drop the `head`, or follow the vendor's docs; and when a conclusion
is "it is structurally impossible", make the search show that rather than merely
failing to find it.

## Admission

`candidate_gate.py` decides from measured latencies: the geometric mean of
`naive_library_composition / expert_dispatch` over the cases where both passed.
A task is not published unless the decision is `admit`.

The mean is geometric on purpose. A task that wins one case tenfold and loses two
at 0.1x is not worth grading, and an arithmetic mean would admit it.

## What the agent sees

The agent-visible tree is the task package as committed. The maintainer side --
the expert dispatch, the gap evidence, the hidden manifest, the environment
snapshot and the admission decision -- lives under `musa_operator_eval/private/`,
which `.gitignore` keeps out of the repository apart from its README. There is no
export step: publishing is what is already committed.

That separation is a claim, so it is tested rather than assumed. `tests/test_private_assets.py`
fails if anything under `private/` other than its README is tracked by git.

## Release gate

Before a task is published:

- every source audit passes;
- public and hidden correctness pass;
- every dispatch trace matches the expected path;
- stability reruns pass;
- every performance measurement comes from the active environment snapshot;
- the run reports `pass`, never `incomplete`.

`incomplete` is a real state rather than a failure: it means a hardware stage has
not produced evidence yet, and it must not be reported as a pass.
