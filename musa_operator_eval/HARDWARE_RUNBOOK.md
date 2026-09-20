# MUSA hardware completion runbook

Run every command in the same immutable image on each admitted target
architecture. Never reuse latency numbers across environments.

## 1. Capture the environment

Run `tools/collect_environment.py` with the exact device, architecture, driver,
Toolkit, muDNN, muBLAS, image digest, architecture flag and fast-math setting.
Store the resulting snapshot next to the task contract; it is the identifier
every later measurement is tied to.

## 2. Probe capabilities

Run `tools/probe_sdpa.py`, then add native muDNN and MATE probes. A high-level
torch_musa success stays `selected_route=unverified` until profiler or direct
adapter evidence identifies the route.

Record at least three real library gaps covering at least two reason
categories. Use those records to fill
`templates/private_gap_evidence.template.json`, then run:

```bash
python musa_operator_eval/tools/make_private_manifest.py \
  --evidence <private-gap-evidence.json> \
  --output musa_operator_eval/private/sdpa_forward_b_v0/cases.private.json
```

## 3. Implement the reference and expert solutions

The reference `Model` stays plain PyTorch and lives in the task's `problem.py`
alongside `get_inputs` / `get_init_inputs`, matching how every other KernelBench
problem is written. The maintainer-side expert dispatch is a `ModelNew`
implementing the `fused_library → library_composition → custom_fallback` order
with runtime probing; build it against the real muDNN / torch_musa APIs and
verify it against the reference before it becomes the speedup denominator.

Run the submission-side audit with the task's own policy:

```python
from kernelbench.kernel_static_checker import validate_library_kernel_static
valid, errors, warnings = validate_library_kernel_static(source, LIBRARY_POLICY)
```

This inverts the A-tier premise: whitelisted library compute is allowed, and the
hard errors become non-whitelisted imports, version-string dispatch, and a
missing dispatch trace.

## 4. Establish baselines

Copy `templates/baseline.b.template.json` into the private task directory and
measure all six implementations in the same image:

1. upstream MUSA implementation;
2. muDNN fused implementation;
3. torch_musa SDPA;
4. torch_musa eager;
5. naive library composition;
6. expert dispatch.

Use MUSA events, 10 warmups, 100 measurements, 5 rounds, and the median unless
the task contract is versioned to change this protocol. Then apply:

```bash
python musa_operator_eval/tools/candidate_gate.py \
  musa_operator_eval/private/sdpa_forward_b_v0/baseline.json \
  --output musa_operator_eval/private/sdpa_forward_b_v0/admission.json
```

Do not publish the task unless the decision is `admit` and the measured ratio
is at least 1.3.

## 5. Run the evaluation

Mount the private directory only into the evaluator. `tools/run_task.py` drives
the whole task: it specializes the reference per case, evaluates the submission,
and checks the dispatch trace.

```bash
python musa_operator_eval/tools/run_task.py \
  --submission <model_new.py> --backend musa --precision fp16 \
  --output runs/sdpa_forward_b_v0/report.json
```

Underneath, correctness, timing and speedup run through the KernelBench
framework:

- `src/kernelbench/eval.py` (`eval_library_dispatch_against_ref`) with `backend="musa"`;
- `num_correct_trials` drives randomized correctness inputs, so evaluation
  inputs are never a fixed public file;
- `src/kernelbench/dispatch_trace.py` supplies the trace transport and check;
- `src/kernelbench/timing.py` supplies the timer;
- `src/kernelbench/kernel_static_checker.py` supplies the tier-aware source audit;
- `scripts/generate_baseline_time.py` supplies the baseline measurement path.

The audit runs before anything is compiled, and the tier is read from the
`TIER` / `LIBRARY_POLICY` names in the problem source, so a B-tier task is
audited with the inverted rule set. A rejected submission never reaches the
device. Every result carries `metadata["tier"]`, `dispatch_trace_passed`, and the
trace summary, errors and path under metadata.

The release gate is:

- all source audits pass;
- public and hidden correctness pass;
- every dispatch trace matches the expected path;
- stability reruns pass;
- all performance measurements come from the active environment snapshot;
- final report status is `pass`, never `incomplete`.

## 6. Publish the task

The evaluator-only tree (`private/`, `sources/`, `capability_plan.json` and the
expert solution) is never copied into the agent-visible checkout; keep it in a
separate repository or access-controlled artifact store and mount it only into
the evaluator. A dedicated exporter is not currently implemented — the previous
one was removed, and its replacement should be a thin copy step plus a lint that
rejects upstream provenance (repository URL, commit hash, source project name)
leaking into the task files.
