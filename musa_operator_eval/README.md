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

The one framework change that lives with this work: a problem file may declare
`TIER` and `LIBRARY_POLICY`, and `eval.py` reads them to pick the matching
source-audit rule set. See `kernel_static_checker.static_audit_kernel`.

## Layout

- `tasks/`: agent-visible task packages. Each carries a KernelBench-format
  `problem.py` (`Model`, `get_inputs`, `get_init_inputs`, plus `TIER` and
  `LIBRARY_POLICY`), a `semantics.json` contract, a `PROMPT.md`, and the case list.
- `agent_reference/`: sanitized, agent-visible capability descriptions.
- `sources/`: provenance and license records for the upstream snapshots a task derives from.
- `templates/`: baseline and gap-evidence templates.
- `tools/`: capability probing, environment capture, case generation, admission gate.
- `tests/`: CPU-only unit tests for the tools above.
- `private/`: evaluator-only assets; ignored by Git except for its README.

## Tools

```bash
# Capture the MUSA environment (device, driver, toolkit, muDNN, image digest)
python musa_operator_eval/tools/collect_environment.py

# Probe fused-library / composition coverage per route
python musa_operator_eval/tools/probe_sdpa.py

# Deterministically regenerate public inputs and golden outputs from case.seed
python musa_operator_eval/tools/generate_cases.py

# Build a private hidden-case manifest from measured gap evidence
python musa_operator_eval/tools/make_private_manifest.py \
  --evidence <private-gap-evidence.json> \
  --output musa_operator_eval/private/<task-id>/cases.private.json

# Apply the B-tier naive/expert admission gate
python musa_operator_eval/tools/candidate_gate.py <baseline.json>

# Drive a task across all of its cases
python musa_operator_eval/tools/run_task.py --submission <model_new.py> --static-only
python musa_operator_eval/tools/run_task.py --submission <model_new.py> \
  --backend musa --precision fp16 --output runs/<task>/report.json
```

`run_task.py` is the entry point for a whole task. It specializes `problem.py`
per case using the `case_parameters` mapping in `task.json`, evaluates the
submission against the reference, and checks the dispatch trace. `--static-only`
stops after the source audit and the case-override check, so it runs on a
machine with no device.

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

# Cross-checks the task's PyTorch reference against the NumPy one (needs torch)
python musa_operator_eval/tests/test_problem_reference.py -v

# Framework-level B-tier support (stdlib only)
python src/kernelbench/unit_tests/test_library_static_checker.py -v
python src/kernelbench/unit_tests/test_dispatch_trace.py -v
```
