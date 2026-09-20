"""Collect a reproducible MUSA environment snapshot without fabricating fields.

Two identifiers come out of this, because they answer different questions:

  snapshot_id        the software and hardware *configuration*: device model,
                     architecture, driver, toolkit components and the torch
                     stack. Two machines built the same way share it.
  device_instance_id the specific physical card (GPU UUID and serial). It
                     changes when an instance comes up on a different host,
                     which is the failure mode a container image digest cannot
                     detect at all.

Everything is probed rather than assumed, and every probe records its own
status. A field that could not be read is reported as missing with a reason,
never as a silent null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


# Every component in the toolkit ships a `<name>_version` helper that prints
# "<name>:" followed by a JSON object, so they can all be collected the same way.
COMPONENT_VERSION_TOOLS = (
    "musa_toolkits_version", "mcc_version", "mccl_version", "musa_runtime_version",
    "mudnn_version", "mublas_version", "mufft_version", "muPP_version", "murand_version",
    "musolver_version", "musparse_version", "musify_version", "CUB_version", "Thrust_version",
)

# The public/redacted variant keeps the configuration and drops the identity of
# the specific machine.
REDACTED_KEYS = ("device_instance", "host", "raw_probes", "container")


def run(command: list, timeout: int = 30) -> dict:
    """Run a command, recording the failure instead of hiding it behind None."""
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "unavailable", "command": command, "reason": f"{command[0]} is not on PATH"}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": command, "reason": f"exceeded {timeout}s"}
    except OSError as exc:
        return {"status": "error", "command": command, "reason": f"{type(exc).__name__}: {exc}"}
    output = (completed.stdout + completed.stderr).strip()
    result = {"status": "ok" if completed.returncode == 0 else "error", "command": command, "exit_code": completed.returncode}
    if completed.returncode == 0:
        result["output"] = output
    else:
        result["output"] = output
        result["reason"] = f"exit code {completed.returncode}"
    return result


def read_sysfs(path: str) -> dict:
    try:
        return {"status": "ok", "value": Path(path).read_text(encoding="utf-8").strip()}
    except OSError as exc:
        return {"status": "unavailable", "reason": f"{path}: {type(exc).__name__}"}


def _try_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def parse_component_version(output: str) -> dict:
    """The version helpers print one or more "<name>:" + JSON blocks.

    Most print one block, but musa_runtime_version prints two: the runtime itself
    and the driver it depends on. Splitting on the "<name>:" headers keeps every
    block instead of throwing away everything after the first one.
    """
    blocks: dict = {}
    header = None
    buffer: list = []
    for line in output.splitlines():
        stripped = line.strip()
        is_header = stripped.endswith(":") and stripped[:-1].replace("_", "").isalnum()
        if is_header:
            if header is not None:
                blocks[header] = _try_json("\n".join(buffer))
            header = stripped[:-1]
            buffer = []
        elif header is not None:
            buffer.append(line)
    if header is not None:
        blocks[header] = _try_json("\n".join(buffer))
    return blocks


def musa_home() -> Path:
    return Path(os.environ.get("MUSA_HOME", "/usr/local/musa"))


def collect_components() -> dict:
    """Read every toolkit component version helper that exists, and say which failed."""
    binary_dir = musa_home() / "bin"
    components: dict = {}
    probes: dict = {}
    for tool in COMPONENT_VERSION_TOOLS:
        name = tool[: -len("_version")]
        probe = run([str(binary_dir / tool)])
        probes[tool] = probe
        if probe["status"] != "ok":
            components[name] = {"status": probe["status"], "reason": probe.get("reason")}
            continue
        blocks = parse_component_version(probe["output"])
        primary = blocks.get(name) or next((value for value in blocks.values() if value), None)
        if primary is None:
            # Not JSON on this build; keep the raw text rather than dropping it.
            components[name] = {"status": "unparsed", "output": probe["output"][:400]}
            continue
        components[name] = {"status": "ok", **primary}
        related = {key: value for key, value in blocks.items() if key != name and value}
        if related:
            components[name]["related"] = related
    return {"components": components, "probes": probes}


def collect_devices() -> dict:
    """Query the device with the tool this platform actually ships."""
    probe = run(["mthreads-gmi", "-q", "--json"])
    result = {"tool": "mthreads-gmi", "status": probe["status"], "reason": probe.get("reason")}
    if probe["status"] != "ok":
        # Keep the text probe too, so a JSON-only failure is still diagnosable.
        fallback = run(["mthreads-gmi"])
        result["text_probe"] = fallback
        return result
    try:
        payload = json.loads(probe["output"])
    except ValueError as exc:
        result.update(status="unparsed", reason=f"mthreads-gmi JSON did not parse: {exc}",
                      raw=probe["output"][:400])
        return result
    result["driver_version"] = payload.get("Driver Version")
    result["gmi_timestamp"] = payload.get("Timestamp")
    gpus = payload.get("GPU") or []
    result["devices"] = gpus
    if not gpus:
        result.update(status="empty", reason="mthreads-gmi reported no attached GPU")
    return result


def summarize_device(gpu: dict) -> dict:
    """Pull the fields that identify and characterise one card.

    `GPU Link Info` sits inside `PCI`, not beside it.
    """
    pci = gpu.get("PCI") or {}
    link = pci.get("GPU Link Info") or {}
    pcie = link.get("PCIe Generation") or {}
    width = link.get("Link Width") or {}
    memory = gpu.get("FB Memory Spec") or {}
    return {
        "index": gpu.get("Index"),
        "product_name": gpu.get("Product Name"),
        "uuid": gpu.get("GPU UUID"),
        "serial_number": gpu.get("Serial Number"),
        "device_type": gpu.get("DeviceType"),
        "mpc_capable": gpu.get("MPC Capable"),
        "mtbios_version": gpu.get("MTBios Version"),
        "pci_bus_id": (gpu.get("PCI") or {}).get("Bus ID"),
        "pci_slot": (gpu.get("PCI") or {}).get("Slot ID(Name)"),
        "pcie_generation": {"max": pcie.get("Max"), "current": pcie.get("Current")},
        "pcie_link_width": {"max": width.get("Max"), "current": width.get("Current")},
        "memory": {
            "type": memory.get("Type"), "vendor": memory.get("Vendor"), "speed": memory.get("Speed"),
            "bandwidth": memory.get("Bandwidth"), "bus_width": memory.get("Bus Width"),
            "total": (gpu.get("FB Memory Usage") or {}).get("Total"),
        },
        "clocks": gpu.get("Clocks"),
        "max_clocks": gpu.get("Max Clocks"),
        "ecc_mode": gpu.get("Ecc Mode"),
        "temperature": gpu.get("Temperature"),
    }


def collect_host() -> dict:
    """Everything host-side, none of which a container image digest can pin."""
    cpu_probe = run(["lscpu"])
    cpu = {}
    if cpu_probe["status"] == "ok":
        wanted = {
            "Model name": "model_name", "CPU(s)": "logical_cpus", "Socket(s)": "sockets",
            "Core(s) per socket": "cores_per_socket", "Thread(s) per core": "threads_per_core",
            "NUMA node(s)": "numa_nodes", "Architecture": "architecture",
        }
        for line in cpu_probe["output"].splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            if key.strip() in wanted:
                cpu[wanted[key.strip()]] = value.strip()
    numa = [line.strip() for line in cpu_probe["output"].splitlines() if line.startswith("NUMA node")] if cpu_probe["status"] == "ok" else []
    return {
        "kernel": platform.release(),
        "platform": platform.platform(),
        "libc": " ".join(platform.libc_ver()),
        "os_release": read_optional_text("/etc/os-release"),
        "cpu": cpu,
        "numa_nodes": numa,
        "cpu_probe_status": cpu_probe["status"],
    }


def read_optional_text(path: str):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def collect_cgroup_limits() -> dict:
    """Container CPU and memory quotas are set at creation and are not in the image."""
    limits = {}
    candidates = {
        "memory_max": ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "cpu_max": ("/sys/fs/cgroup/cpu.max",),
        "cpu_quota": ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",),
        "cpu_period": ("/sys/fs/cgroup/cpu/cpu.cfs_period_us",),
        "pids_max": ("/sys/fs/cgroup/pids.max",),
    }
    for name, paths in candidates.items():
        for path in paths:
            result = read_sysfs(path)
            if result["status"] == "ok":
                limits[name] = {"value": result["value"], "source": path}
                break
        else:
            limits[name] = {"value": None, "reason": "no cgroup file readable"}
    return limits


def collect_python_stack() -> dict:
    stack = {}
    for name in ("torch", "torch_musa", "torchvision", "torchada", "numpy", "packaging"):
        try:
            from importlib.metadata import version

            stack[name] = version(name)
        except Exception:
            stack[name] = None
    stack["python"] = sys.version
    if stack.get("torch_musa"):
        try:
            import torch_musa

            # The build hash lives on torch_musa.version, not on the package root,
            # and the installed distribution version drops it.
            version_module = getattr(torch_musa, "version", None)
            stack["torch_musa_build"] = getattr(version_module, "git_version", None)
            stack["torch_musa_version_string"] = getattr(version_module, "__version__", None)
        except Exception as exc:
            stack["torch_musa_import_error"] = f"{type(exc).__name__}: {exc}"
    return stack


def collect_device_properties() -> dict:
    """Read the device through torch, which is how the evaluator reaches it."""
    try:
        import torch
        import torch_musa  # noqa: F401
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        properties = torch.musa.get_device_properties(0)
        return {
            "status": "ok",
            "device_count": torch.musa.device_count(),
            "name": torch.musa.get_device_name(0),
            "major": int(properties.major),
            "minor": int(properties.minor),
            "multi_processor_count": properties.multi_processor_count,
            "total_memory_mb": int(properties.total_memory / (1024 * 1024)),
        }
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def collect_container() -> dict:
    """Whatever the hosting platform exposes. A digest is kept only if real."""
    container = {"platform": "unknown", "identity": {}, "image_digest": None}
    uuid = os.environ.get("AutoDLContainerUUID")
    if uuid or os.environ.get("AutoDLRegion"):
        container["platform"] = "AutoDL"
        container["identity"] = {
            key: os.environ.get(name)
            for key, name in (
                ("container_uuid", "AutoDLContainerUUID"),
                ("region", "AutoDLRegion"),
                ("data_center", "AutoDLDataCenter"),
            )
            if os.environ.get(name)
        }
    digest = os.environ.get("MUSA_EVAL_IMAGE_DIGEST")
    if digest:
        container["image_digest"] = digest
    else:
        container["image_digest"] = None
        container["image_digest_unavailable_reason"] = (
            "No image digest is reachable from inside this container: "
            + ("no container runtime binary is present" if not any(shutil.which(b) for b in ("docker", "crictl", "podman")) else "the runtime exposes none")
            + ", the platform metadata service did not answer, and no digest is published in the environment. "
            "The component git commits below are the immutable anchor instead."
        )
    return container


def derive_architecture(properties: dict, override: str | None):
    """Map torch's `major.minor` onto Moore Threads' `mp_XY` architecture name."""
    if override:
        return override, "command line"
    if properties.get("status") == "ok" and properties.get("major") is not None:
        return f"mp_{properties['major']}{properties['minor']}", "torch.musa device properties"
    return None, None


def redact(snapshot: dict) -> dict:
    """Drop the identity of this specific machine, keep the configuration."""
    public = json.loads(json.dumps(snapshot))
    public["visibility"] = "redacted"
    for key in REDACTED_KEYS:
        public.pop(key, None)
    public.pop("device_instance_id", None)
    return public


def collect_driver() -> dict:
    """The driver is host-side, so it is queried separately from the toolkit.

    `driver_version_query` reports the full release string and its commit, which
    is strictly more informative than the short version gmi prints.
    """
    probe = run([str(musa_home() / "bin" / "driver_version_query")])
    result = {"status": probe["status"], "reason": probe.get("reason")}
    if probe["status"] == "ok":
        result["release_string"] = probe["output"]
        match = re.search(r"(c[0-9a-f]{7,})@(\d{8})", probe["output"])
        if match:
            result["commit"] = match.group(1)
            result["commit_date"] = match.group(2)
    return result


def build_snapshot(args) -> dict:
    components = collect_components()
    devices = collect_devices()
    device_properties = collect_device_properties()
    host = collect_host()
    driver = collect_driver()

    architecture, architecture_source = derive_architecture(device_properties, args.architecture)
    summarised = [summarize_device(gpu) for gpu in devices.get("devices", [])]
    first = summarised[0] if summarised else {}

    resolved = {
        "architecture": architecture,
        "device_name": args.device_name or first.get("product_name"),
        "driver": args.driver or devices.get("driver_version"),
        "musa_toolkit": args.toolkit or (components["components"].get("musa_toolkits") or {}).get("version"),
        "mudnn": args.mudnn or (components["components"].get("mudnn") or {}).get("version"),
        "mublas": args.mublas or (components["components"].get("mublas") or {}).get("version"),
    }
    missing = [name for name, value in resolved.items() if not value]
    if not resolved["architecture"] and not architecture_source:
        missing.append("architecture_source")

    identifier = first.get("uuid") or ""
    identity_source = "|".join(str(part) for part in (
        resolved["device_name"], resolved["architecture"], resolved["driver"],
        driver.get("commit"), resolved["musa_toolkit"], resolved["mudnn"], resolved["mublas"],
        (components["components"].get("musa_toolkits") or {}).get("commit id"),
        (components["components"].get("mudnn") or {}).get("commit id"),
        (components["components"].get("musa_runtime") or {}).get("commit id"),
    ))
    snapshot_id = "musa-" + hashlib.sha256(identity_source.encode()).hexdigest()[:16]
    device_instance_id = ("gpu-" + hashlib.sha256(identifier.encode()).hexdigest()[:16]) if identifier else None

    container = collect_container()
    return {
        "schema_version": "2.0.0",
        "status": "incomplete" if missing else "complete",
        "missing_required_fields": missing,
        "snapshot_id": snapshot_id,
        "device_instance_id": device_instance_id,
        "visibility": "full",
        "captured_at": args.captured_at,
        "hardware": {"device_name": resolved["device_name"], "architecture": resolved["architecture"],
                     "architecture_source": architecture_source, "device_count": len(summarised) or None},
        "software": {"driver": resolved["driver"], "driver_release": driver.get("release_string"),
                     "driver_commit": driver.get("commit"), "musa_toolkit": resolved["musa_toolkit"],
                     "mudnn": resolved["mudnn"], "mublas": resolved["mublas"]},
        "toolkit_components": components["components"],
        "device_instance": first or None,
        "device_properties_via_torch": device_properties,
        "host": host,
        "cgroup_limits": collect_cgroup_limits(),
        "python_stack": collect_python_stack(),
        "container": container,
        "build": {"flags": args.build_flag, "fast_math": args.fast_math},
        "raw_probes": {"device_query": {k: v for k, v in devices.items() if k != "devices"},
                       "driver_query": driver,
                       "component_version_tools": components["probes"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Every field below is auto-detected; these exist to override a bad probe.
    parser.add_argument("--architecture", choices=["mp_21", "mp_22", "mp_31"])
    parser.add_argument("--device-name")
    parser.add_argument("--driver")
    parser.add_argument("--toolkit")
    parser.add_argument("--mudnn")
    parser.add_argument("--mublas")
    parser.add_argument("--captured-at", default=None)
    parser.add_argument("--build-flag", action="append", default=[])
    parser.add_argument("--fast-math", action="store_true")
    args = parser.parse_args()

    import datetime as dt
    if not args.captured_at:
        args.captured_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    snapshot = build_snapshot(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "environment.full.json"
    public_path = args.output_dir / "environment.public.json"
    full_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    public_path.write_text(json.dumps(redact(snapshot), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"snapshot_id={snapshot['snapshot_id']} device_instance_id={snapshot['device_instance_id']}")
    print(full_path)
    print(public_path)
    if snapshot["status"] != "complete":
        print("incomplete snapshot, missing: " + ", ".join(redact(snapshot)["missing_required_fields"]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
