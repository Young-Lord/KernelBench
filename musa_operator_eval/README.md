# MUSA Operator Evaluation

Task contracts, capability evidence and admission tooling for the MUSA A/B
operator evaluation tracks. Version 0 is scoped to Attention families; other
families must add their own semantic contract instead of reusing
Attention-specific fields.

Evaluation itself (compilation, correctness, timing, speedup) is **not**
implemented here. It runs on the KernelBench framework in `src/kernelbench/`:
`eval.py`, `timing.py`, `kernel_static_checker.py`, `musa_extension.py` and
`scripts/generate_baseline_time.py`. This directory only carries what the
framework has no notion of.

Both tracks are packaged, but around different things.

An A-tier task *is* a KernelBench problem: the repository holds 107 of them under
`KernelBench/level1_musa/` and `KernelBench/level3_musa/`, each with its
hand-written answer, and `sets/attention/` records which of them belong to the
attention family and what has been measured on them. An A-tier package does not
redefine that problem — it carries a copy of it, verified byte for byte by a test,
and adds the contract around it: what the submission may not call, at which
precision it is graded, and which cases it is shown.

A B-tier task needs something the framework has no notion of, because it is graded
on how the answer was reached rather than only on what it computes: a library
whitelist, a dispatch order, and a trace the submission has to emit. Its package
carries that contract. `sdpa_forward_pilot/` is the family-scoped form of it — its
problem generalises the attention core past what any one KernelBench entry asks
for, so it names the entries it covers instead of claiming one.

Task directories are named `<set entry>_a` and `<set entry>_b`, and each package
declares the entry it realises under `set_entry`. `tests/test_task_packages.py`
checks that every entry the set marks eligible has a package for that tier, and
that the two tiers of one entry share a semantic contract.


The one framework change that lives with this work: a problem file may declare
`TIER` and `LIBRARY_POLICY`, and `eval.py` reads them to pick the matching
source-audit rule set. See `kernel_static_checker.static_audit_kernel`.

The A tier adds one check to the shipped defaults: `attention_entry_point`, which
rejects a call to a fused attention entry point. That is not because every torch
op is forbidden — it is not, and the answers in `KernelBench/level3_musa/` hold
their projections as `nn.Linear` while computing the attention themselves. It is
because the reference implementation calls that same entry point, so a
submission that calls it matches the reference exactly and passes every
correctness check, while scoring about 2.5x on `kb_l1_97` against a hand-written
baseline the library call beats. The upstream `torch_computation_ops` check that
would also catch it ships as a warning.

Where that check cannot reach, each A-tier contract declares its own
`kernel_scope`: what must be in a kernel, and what may stay in the library. An
`nn.Linear` holding weights and one computing a projection are the same source,
so that half of the boundary is stated for a reader rather than enforced by a
pattern.

## Layout

- `tasks/`: agent-visible task packages. Each carries a KernelBench-format
  `problem.py` (`Model`, `get_inputs`, `get_init_inputs`, plus `TIER` and
  `LIBRARY_POLICY` for the B tier), a `semantics.json` contract, a `PROMPT.md`,
  the public case list, and a `tensor_contract` naming the dtypes it is graded
  at. A task names the environment it was measured on rather than describing it.
  `problem.py` declares its own tier, so the driver refuses to run it at a
  precision the contract does not permit.
- `environments/`: one agent-visible record per machine configuration, named
  after its `snapshot_id`. A task points here; the device model, architecture and
  component versions live here and nowhere else, so a machine is described once
  and two tasks on it cannot disagree about what it is. Full snapshots, which
  keep the serial number and GPU UUID, live under `private/environments/`.
- `sets/`: curated operator sets, each with a hand-written manifest and a
  generated index. `sets/attention/` covers the attention family and records
  which A-tier answer, which measurement and which candidate kernel already
  exist for every entry. A set is a maintainer ledger — it names the A-tier
  answer path and the candidate implementations, so it is never agent-visible.
- `agent_reference/`: sanitized, agent-visible capability descriptions, one file
  per family, named `<family>_capabilities.json`. A capability boundary belongs
  to the library and the operation rather than to one task, so every task in a
  family names the same file; the name is what stops a second family from
  quietly inheriting a first family's boundaries.
- `sources/`: provenance and license records. `attention-source-001.json` covers the upstream snapshots a task derives from; `attention-kernel-candidates.json` inventories candidate attention kernel implementations and which target each can serve.
- `templates/`: baseline and gap-evidence templates.
- `tools/`: capability probing, environment capture, case generation, admission gate, task driver, set verification.
- `tests/`: CPU-only unit tests for the tools above.
- `private/`: evaluator-only assets; ignored by Git except for its README.

## Tools

```bash
# Capture the MUSA environment: device identity, driver, every toolkit component
# with its commit, and the host side no container image pins
python musa_operator_eval/tools/collect_environment.py

# Probe fused-library / composition coverage per route
python musa_operator_eval/tools/probe_sdpa.py

# Deterministically regenerate public inputs and golden outputs from case.seed
python musa_operator_eval/tools/generate_cases.py

# Build a private hidden-case manifest from measured gap evidence
python musa_operator_eval/tools/make_private_manifest.py \
  --evidence <private-gap-evidence.json> \
  --output musa_operator_eval/private/<task-id>/cases.private.json

# Measure the B-tier baselines in the gate's own shape (needs the device)
python musa_operator_eval/tools/measure_baseline.py \
  --task-dir musa_operator_eval/tasks/kb_l3_43_b \
  --naive  musa_operator_eval/private/<task-id>/naive/model_new.py \
  --expert musa_operator_eval/private/<task-id>/expert/model_new.py \
  --blind  musa_operator_eval/private/<task-id>/blind/model_new.py \
  --unavailable musa_operator_eval/private/<task-id>/unavailable.json \
  --output musa_operator_eval/private/<task-id>/baseline.json

# Apply the B-tier naive/expert admission gate
python musa_operator_eval/tools/candidate_gate.py <baseline.json>

# Drive a task across all of its cases
python musa_operator_eval/tools/run_task.py --submission <model_new.py> --static-only
python musa_operator_eval/tools/run_task.py --submission <model_new.py> \
  --backend musa --precision fp16 --output runs/<task>/report.json
# Grade the hidden set, which lives outside the task package
python musa_operator_eval/tools/run_task.py --submission <model_new.py> \
  --cases musa_operator_eval/private/<task-id>/cases.private.json

# Verify a curated set and regenerate its index
python musa_operator_eval/tools/collect_attention_set.py
python musa_operator_eval/tools/collect_attention_set.py --check   # verify only, for CI
```

`run_task.py` is the entry point for a whole task. It specializes `problem.py`
per case using the `case_parameters` mapping in `task.json`, evaluates the
submission against the reference, and checks the dispatch trace. `--static-only`
stops after the source audit and the case-override check, so it runs on a
machine with no device, and `--cases` points it at a manifest outside the task
package, which is how the hidden set is graded without being readable from it.

`measure_baseline.py` measures each implementation per case under the protocol in
its own `PROTOCOL` and writes the shape `candidate_gate.py` reads. Both live on
the device side: they check correctness before timing anything, and they record
an implementation it cannot honestly build as `unavailable` with a reason rather
than substituting a lookalike.

Every implementation is passed in as a module path rather than named in the
tool's source, because reaching a task's attention core is the task's business:
`kb_l3_43_b` takes one input and five init inputs and computes the core behind its
own projections, while the family-scoped pilot takes `q, k, v` directly. The
tool's own contribution is to reproduce the harness's three measurement facts
rather than paraphrase them — weights are aligned by re-seeding and rebuilding as
`kernelbench.eval` does, inputs and parameters are cast by its
`_process_input_tensor`, and correctness is judged by its
`get_tolerance_for_precision`. A baseline measured under a different protocol from
the submissions it is the denominator for would admit a task for the wrong reason.

`collect_attention_set.py` reads `sets/attention/attention_set.json`, which is
curated by hand, and verifies every claim it makes against the repository: the
reference and A-tier answer files exist, and every recorded measurement still
matches the file it was read from. It then regenerates `sets/attention/SET.md`.
Exit code is non-zero on drift, so a re-run of a baseline cannot silently leave
a stale number in the index.

Public inputs and goldens are regenerated on demand and are not committed.
All tools are standard-library-plus-NumPy and are hardware-independent; the
probes and baselines are not, and must run on the target device.

## Tests

```bash
# Case generation, environment and capability tooling (stdlib + NumPy)
python musa_operator_eval/tests/test_environment_tools.py -v
python musa_operator_eval/tests/test_reference_and_cases.py -v

# Task driver: case specialization, static report, CLI (stdlib only)
python musa_operator_eval/tests/test_run_task.py -v

# Task packages: set coverage, A-tier reference identity, contracts, cases, and
# that the driver honours the precision a task's tensor contract permits
python musa_operator_eval/tests/test_task_packages.py -v

# Attention set: measurement drift, generated index well-formedness (stdlib only)
python musa_operator_eval/tests/test_attention_set.py -v

# Admission gate: pairing rules, and that the mean is geometric (stdlib only)
python musa_operator_eval/tests/test_candidate_gate.py -v

# Publication separation: private assets must stay untracked (stdlib only)
python musa_operator_eval/tests/test_private_assets.py -v

# Cross-checks the task's PyTorch reference against the NumPy one (needs torch)
python musa_operator_eval/tests/test_problem_reference.py -v

# Framework-level B-tier support (stdlib only)
python src/kernelbench/unit_tests/test_library_static_checker.py -v
python src/kernelbench/unit_tests/test_dispatch_trace.py -v
```
