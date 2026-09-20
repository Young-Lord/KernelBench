# Environments

One record per machine configuration the evaluation has been measured on. A task
names the record it was measured against and leaves the device model,
architecture, toolkit and library versions here, so a machine is described once
and two tasks on the same machine cannot disagree about what the machine is.

This directory is **agent-visible**. Everything in it is the redacted projection
of a captured snapshot: the configuration, without the serial number, GPU UUID or
hostname of the specific card it was captured on. The full snapshot keeps that
identity and lives under `private/environments/`, which is not published.

## Two identifiers

A snapshot carries two, and they answer different questions:

- `snapshot_id` identifies the **configuration**: device model, architecture,
  MUSA toolkit, driver, muDNN/muBLAS, torch_musa build, Python stack. It is
  derived from those facts, so two cards with the same configuration share it.
- `device_instance_id` identifies the **physical card**, derived from its GPU
  UUID. It is removed from the public record on purpose.

Latencies may only be compared between runs whose `snapshot_id` matches. A
difference in `device_instance_id` under the same `snapshot_id` is a different
card with the same configuration; treat it as reusable only when the instance is
known to be equivalent.

## Available

| snapshot_id | device | MUSA arch | dtype | toolkit | muDNN | muBLAS | driver |
|---|---|---|---|---|---|---|---|
| `musa-5f9d7b9dd1233a68` | MTT S4000 | `mp_22` | fp16 / bf16 | 3.1.0 | 2.7.0 | 1.6.0 | 20241025 kuae1.3.0_musa3.1.0 |

Which tasks were measured here:

| task | tier | record |
|---|---|---|
| `sdpa_forward_b_v0` | B_library | `musa-5f9d7b9dd1233a68.public.json` |

## The records

### `musa-5f9d7b9dd1233a68.public.json`

MTT S4000, `mp_22`. The machine every attention measurement in this repository
was taken on: the A-tier hand-written kernels in `sets/attention/`, the B-tier
admission baseline for `sdpa_forward_b_v0`, and the SDPA route probe.

Two facts about it constrain what any task here can ask for:

- The image ships library builds for `mp_21` and `mp_22` only. There is no
  `mp_31` build, so an S5000 number cannot be produced from this machine.
- `mp_22` uses 128-thread warps; `mp_31` uses 32. Every tuned A-tier launch
  configuration in this repository was tuned at 128, so none of them transfer to
  an S5000 target without re-measuring.

## Adding one

Capture it on the machine, which writes both projections:

```bash
python musa_operator_eval/tools/collect_environment.py --output-dir <dir>
```

Then move `environment.public.json` here as `<snapshot_id>.public.json` and the
full one to `private/environments/<snapshot_id>.full.json`. The tests check that
every record a task names resolves here, that the name matches the `snapshot_id`
inside it, and that nothing the collector redacts survived into it.
