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

## Layout

- `tasks/`: agent-visible task packages (contract, semantics, prompt, cases, reference).
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
```

Public inputs and goldens are regenerated on demand and are not committed.
All tools are standard-library-plus-NumPy and are hardware-independent; the
probes and baselines are not, and must run on the target device.

## Tests

```bash
python musa_operator_eval/tests/test_environment_tools.py -v
python musa_operator_eval/tests/test_reference_and_cases.py -v
```
