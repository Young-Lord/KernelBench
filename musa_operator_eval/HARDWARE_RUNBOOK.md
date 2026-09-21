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

## Three traps worth knowing before reading a number

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
