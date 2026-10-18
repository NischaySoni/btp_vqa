#!/usr/bin/env python3
"""Collect a shareable report about CUDA visibility and basic PyTorch GPU usability.

This script does not change CUDA environment variables, reset devices, or install packages.
It tests only devices visible to this process (as controlled by CUDA_VISIBLE_DEVICES).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}", flush=True)


def run_command(args: list[str], timeout: int = 30) -> None:
    print(f"$ {' '.join(args)}", flush=True)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        print(f"exit_code: {result.returncode}")
        print("--- stdout ---")
        print(result.stdout.rstrip() or "<empty>")
        print("--- stderr ---")
        print(result.stderr.rstrip() or "<empty>")
    except FileNotFoundError:
        print(f"command not found: {args[0]}")
    except subprocess.TimeoutExpired as exc:
        print(f"command timed out after {timeout}s")
        if exc.stdout:
            print("partial stdout:")
            print(exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout)
        if exc.stderr:
            print("partial stderr:")
            print(exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr)
    except Exception:
        traceback.print_exc()


def main() -> int:
    print("CUDA diagnostic report", flush=True)
    print("timestamp_utc:", datetime.now(timezone.utc).isoformat(), flush=True)

    section("process and environment")
    print("python_executable:", sys.executable)
    print("python_version:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        print(f"{name}:", os.environ.get(name, "<unset>"))

    section("NVIDIA driver utilities")
    run_command(["nvidia-smi", "-L"])
    run_command([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu",
        "--format=csv",
    ])

    section("NVIDIA device nodes")
    nodes = sorted(Path("/dev").glob("nvidia*"))
    if not nodes:
        print("No /dev/nvidia* device nodes found")
    for node in nodes:
        try:
            st = node.stat()
            print(f"{node}: mode={oct(st.st_mode & 0o777)} rdev={st.st_rdev}")
        except Exception:
            print(f"{node}: stat failed")
            traceback.print_exc()

    section("PyTorch CUDA")
    try:
        import torch
    except Exception:
        print("PyTorch import failed:")
        traceback.print_exc()
        return 1

    print("torch_version:", torch.__version__)
    print("torch_file:", torch.__file__)
    print("torch_cuda_build:", torch.version.cuda)
    print("torch_hip_build:", torch.version.hip)

    try:
        available = torch.cuda.is_available()
        print("torch.cuda.is_available():", available)
    except Exception:
        available = False
        print("torch.cuda.is_available() raised:")
        traceback.print_exc()

    try:
        count = torch.cuda.device_count()
        print("torch.cuda.device_count():", count)
    except Exception:
        count = 0
        print("torch.cuda.device_count() raised:")
        traceback.print_exc()

    try:
        torch.cuda.init()
        print("torch.cuda.init(): SUCCESS")
    except Exception:
        print("torch.cuda.init(): FAILED")
        traceback.print_exc()

    usable = 0
    for index in range(count):
        print(f"\nvisible_device_{index}:", flush=True)
        try:
            props = torch.cuda.get_device_properties(index)
            print("name:", props.name)
            print("total_memory_bytes:", props.total_memory)
            print("compute_capability:", f"{props.major}.{props.minor}")
            x = torch.arange(256, dtype=torch.float32, device=f"cuda:{index}").reshape(16, 16)
            y = x @ x
            torch.cuda.synchronize(index)
            print("small_tensor_matmul_and_sync:", "SUCCESS", "result_sum:", y.sum().item())
            usable += 1
        except Exception:
            print("device_test: FAILED")
            traceback.print_exc()

    section("summary")
    print("visible_devices:", count)
    print("usable_devices_passed_tensor_test:", usable)
    if usable:
        print("result: at least one visible CUDA device completed a PyTorch operation")
        return 0
    print("result: no visible CUDA device completed the PyTorch tensor test")
    print("Note: an unset or restrictive CUDA_VISIBLE_DEVICES can hide host GPUs from this process.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
