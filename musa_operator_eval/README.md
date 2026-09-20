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

The two tracks are not packaged the same way, because they do not need the same
things. An A-tier task *is* a KernelBench problem: the repository already holds
107 of them under `KernelBench/level1_musa/` and `KernelBench/level3_musa/`, each
with its hand-written answer, and `sets/attention/` records which of them belong
to the attention family and what has been measured on them. Nothing here
redefines them. A B-tier task does need a package, because a tier that is graded
on how the answer was reached has to state that contract somewhere, and only
`tasks/sdpa_forward_pilot/` does.

The one framework change that lives with this work: a problem file may declare
`TIER` and `LIBRARY_POLICY`, and `eval.py` reads them to pick the matching
source-audit rule set. See `kernel_static_checker.static_audit_kernel`.

## Layout

- `tasks/`: agent-visible task packages. Each carries a KernelBench-format
  `problem.py` (`Model`, `get_inputs`, `get_init_inputs`, plus `TIER` and
  `LIBRARY_POLICY`), a `semantics.json` contract, a `PROMPT.md`, and the case list.
  Only B-tier tasks need one.
- `sets/`: curated operator sets, each with a hand-written manifest and a
  generated index. `sets/attention/` covers the attention family and records
  which A-tier answer, which measurement and which candidate kernel already
  exist for every entry.
- `agent_reference/`: sanitized, agent-visible capability descriptions.
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
  --task-dir <task> --expert <model_new.py> --output <baseline.json>

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
