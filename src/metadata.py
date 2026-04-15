"""Utilities for metadata collection and JSON serialization."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import torch



def collect_environment_metadata() -> Dict[str, Any]:
    """Collect environment metadata useful for reproducibility.

    This metadata is intentionally verbose so that different runs can be audited later.
    """
    metadata: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "pid": os.getpid(),
    }

    if torch.cuda.is_available():
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_devices"] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            metadata["cuda_devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    return metadata



def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a Python dictionary as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
