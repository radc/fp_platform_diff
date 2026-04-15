"""Input generation stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from .io_utils import get_torch_dtype, save_tensor_any
from .metadata import collect_environment_metadata, save_json



def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)



def _make_tensor(shape: List[int], dtype: torch.dtype, config: Dict[str, Any]) -> torch.Tensor:
    """Create a tensor based on the configured distribution."""
    distribution = config["distribution"]

    if distribution == "uniform":
        low = float(config["uniform_low"])
        high = float(config["uniform_high"])
        tensor = (high - low) * torch.rand(shape, dtype=torch.float32) + low
    elif distribution == "normal":
        mean = float(config["normal_mean"])
        std = float(config["normal_std"])
        tensor = torch.randn(shape, dtype=torch.float32) * std + mean
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return tensor.to(dtype)



def generate_inputs_from_config(config_path: Path) -> None:
    """Generate input tensors from a JSON config file."""
    config = _load_config(config_path)

    experiment_name = config["experiment_name"]
    output_root = Path(config["output_root"])
    experiment_root = output_root / experiment_name
    input_root = experiment_root / "inputs"
    metadata_root = experiment_root / "metadata"

    input_cfg = config["inputs"]
    seed = int(config["seed"])
    torch.manual_seed(seed)

    tensor_names = input_cfg.get("tensor_names")
    num_tensors = int(input_cfg["num_tensors"])
    if tensor_names is None:
        tensor_names = [chr(ord("a") + i) for i in range(num_tensors)]
    if len(tensor_names) != num_tensors:
        raise ValueError("tensor_names length must match num_tensors")

    shape = list(input_cfg["shape"])
    num_elements = int(input_cfg["num_elements"])
    inferred_elements = 1
    for dim in shape:
        inferred_elements *= int(dim)
    if inferred_elements != num_elements:
        raise ValueError(
            f"shape product ({inferred_elements}) does not match num_elements ({num_elements})"
        )

    dtype_name = input_cfg["dtype"]
    dtype = get_torch_dtype(dtype_name)
    save_formats = list(input_cfg["save_formats"])

    generation_metadata: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "seed": seed,
        "shape": shape,
        "num_elements": num_elements,
        "dtype": dtype_name,
        "tensor_names": tensor_names,
        "save_formats": save_formats,
        "distribution": input_cfg["distribution"],
        "environment": collect_environment_metadata(),
    }

    for tensor_name in tensor_names:
        tensor = _make_tensor(shape=shape, dtype=dtype, config=input_cfg)
        save_tensor_any(
            tensor=tensor,
            base_path_without_suffix=input_root / tensor_name,
            formats=save_formats,
            dtype_name=dtype_name,
        )

    save_json(generation_metadata, metadata_root / "generation_metadata.json")
    print(f"Input tensors generated in: {input_root}")
