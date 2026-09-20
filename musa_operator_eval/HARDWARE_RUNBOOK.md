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
python musa_operator_eval/tools/run_task.py \
  --cases musa_operator_eval/private/sdpa_forward_b_v0/cases.private.json \
  --submission <model_new.py> --backend musa --precision fp16 \
  --output runs/sdpa_forward_b_v0/hidden_report.json
```

The hidden manifest is built by `make_private_manifest.py` from recorded gap
evidence. It refuses to build a manifest that does not cover at least
`minimum_gap_cases` gaps across at least `required_gap_reasons` categories, both
read from the task contract.

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
