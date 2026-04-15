"""Execution stage: load inputs, run operation.py, save step outputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import torch

from .io_utils import load_tensor_any
from .metadata import collect_environment_metadata, save_json
from .ops_context import OperationContext



def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)



def _load_operation_module(operation_file: Path):
    """Dynamically import the user-provided operation.py file."""
    spec = importlib.util.spec_from_file_location("user_operation_module", operation_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load operation module from {operation_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def execute_from_config(
    config_path: Path,
    device: str,
    run_name: str,
    operation_file: Path,
) -> None:
    """Execute the configured operation script on a selected device."""
    config = _load_config(config_path)

    experiment_name = config["experiment_name"]
    output_root = Path(config["output_root"])
    experiment_root = output_root / experiment_name
    input_root = experiment_root / "inputs"
    execution_root = experiment_root / "executions" / run_name
    steps_dir = execution_root / "steps"
    final_dir = execution_root / "final"
    metadata_dir = execution_root / "metadata"

    input_cfg = config["inputs"]
    exec_cfg = config["execution"]

    dtype_name = input_cfg["dtype"]
    tensor_names = input_cfg["tensor_names"]
    shape = input_cfg["shape"]
    load_format = exec_cfg["load_format"]
    save_step_formats = exec_cfg["save_step_formats"]
    save_final_formats = exec_cfg["save_final_formats"]

    deterministic_algorithms = bool(exec_cfg.get("deterministic_algorithms", False))
    disable_tf32 = bool(exec_cfg.get("disable_tf32", True))

    torch.use_deterministic_algorithms(deterministic_algorithms)

    if disable_tf32:
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available")

    loaded_inputs = {}
    for tensor_name in tensor_names:
        tensor = load_tensor_any(
            base_path_without_suffix=input_root / tensor_name,
            fmt=load_format,
            dtype_name=dtype_name,
            shape=shape,
            map_location="cpu",
        )
        loaded_inputs[tensor_name] = tensor.to(device)

    ctx = OperationContext(
        inputs=loaded_inputs,
        dtype_name=dtype_name,
        save_step_formats=save_step_formats,
        save_final_formats=save_final_formats,
        steps_dir=steps_dir,
        final_dir=final_dir,
    )

    module = _load_operation_module(operation_file)
    if not hasattr(module, "run"):
        raise AttributeError(f"The operation file {operation_file} does not define run(ctx)")

    outputs = module.run(ctx)
    if not isinstance(outputs, dict):
        raise TypeError("run(ctx) must return a dictionary of final outputs")

    ctx.save_final_outputs(outputs)

    execution_metadata = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "device": device,
        "dtype": dtype_name,
        "shape": shape,
        "load_format": load_format,
        "save_step_formats": save_step_formats,
        "save_final_formats": save_final_formats,
        "deterministic_algorithms": deterministic_algorithms,
        "disable_tf32": disable_tf32,
        "step_records": ctx.step_records,
        "environment": collect_environment_metadata(),
        "operation_file": str(operation_file),
    }
    save_json(execution_metadata, metadata_dir / "execution_metadata.json")
    print(f"Execution finished. Results stored in: {execution_root}")
