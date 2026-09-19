# MUSA hardware completion runbook

Run every command in the same immutable image on each admitted target architecture.
Never reuse latency numbers across environments.

## 1. Capture the environment

Run `tools/collect_environment.py` with the exact device, architecture, driver,
Toolkit, muDNN, muBLAS, image digest, architecture flag, and fast-math setting.
Validate the resulting public snapshot with `schemas/environment.schema.json`.

## 2. Probe capabilities

Run `tools/probe_sdpa.py`, then add native muDNN and MATE probes. A high-level
torch_musa success remains `selected_route=unverified` until profiler or direct
adapter evidence identifies the route.

Record at least three real library gaps covering at least two reason categories.
Use those records to fill `templates/private_gap_evidence.template.json`, then run:

```bash
python musa_operator_eval/tools/make_private_manifest.py \
  --evidence <private-gap-evidence.json> \
  --output musa_operator_eval/private/sdpa_forward_b_v0/cases.private.json
```

## 3. Implement and build native adapters

Implement runtime probing and dispatch only inside the marked Starter regions.
Build separately for `mp_22` and `mp_31`. Record actual linker inputs and inspect
the final binary dependency/import table against `starter_manifest.json`.

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

Do not publish the task unless the decision is `admit` and the measured ratio is
at least 1.3.

## 5. Run the evaluator

Mount the private directory only into the evaluator, then supply the compiled
runner and environment snapshot to `evaluator/evaluate.py`. The release gate is:

- all Schema and source audits pass;
- public and hidden correctness pass;
- every dispatch trace matches the expected path;
- stability reruns pass;
- all performance measurements come from the active environment snapshot;
- final report status is `pass`, never `incomplete`.

## 6. Export the public task

After the private release gate passes, export into a new empty directory:

```bash
python musa_operator_eval/tools/package_public_task.py --output <release-dir>
```

Inspect the release directory before publication. It must not contain capability
evidence, hidden cases, source snapshots, expert source, or baseline artifacts.
