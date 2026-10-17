"""Resource snapshots. RSS is this process; GPU peaks are PyTorch allocations."""

import os
import platform
import subprocess
import time

import torch


def resources():
    result = {"cpu_seconds": time.process_time()}
    try:
        import psutil

        process = psutil.Process()
        result.update(rss_bytes=process.memory_info().rss, cpu_seconds=sum(process.cpu_times()[:2]))
    except ImportError:
        result.update(rss_bytes=None)
    if torch.cuda.is_available():
        result.update(
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return result


def hardware():
    result = dict(
        cpu=platform.processor(),
        logical_cpus=os.cpu_count(),
        platform=platform.platform(),
        torch_cpu_threads=torch.get_num_threads(),
    )
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            gpu_total_bytes=properties.total_memory, gpu_compute_capability=list(torch.cuda.get_device_capability())
        )
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["driver_version"] = r.stdout.strip() if r.returncode == 0 else None
    return result
