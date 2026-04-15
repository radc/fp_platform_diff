"""Tensor I/O helpers for PT, BIN, and TXT formats."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


NUMPY_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
}



def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a string dtype name to a PyTorch dtype."""
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_MAP[dtype_name]



def save_tensor_pt(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor in PyTorch native format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)



def load_tensor_pt(path: Path, map_location: str = "cpu") -> torch.Tensor:
    """Load a tensor from PyTorch native format."""
    tensor = torch.load(path, map_location=map_location)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor)}")
    return tensor



def save_tensor_bin(tensor: torch.Tensor, path: Path, dtype_name: str) -> None:
    """Save a tensor as raw binary bytes.

    The caller must store shape and dtype in metadata, because raw binary does not carry schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensor = tensor.detach().cpu()

    if dtype_name == "bfloat16":
        # NumPy has limited support for bfloat16 portability across environments.
        # To avoid ambiguity, save bfloat16 data through uint16 bit representation.
        uint16_view = cpu_tensor.view(torch.uint16).numpy()
        uint16_view.tofile(path)
        return

    np_dtype = NUMPY_DTYPE_MAP.get(dtype_name)
    if np_dtype is None:
        raise ValueError(f"Binary save is not implemented for dtype {dtype_name}")
    cpu_tensor.numpy().astype(np_dtype, copy=False).tofile(path)



def load_tensor_bin(path: Path, dtype_name: str, shape: Sequence[int]) -> torch.Tensor:
    """Load a tensor from a raw binary file using explicit dtype and shape."""
    if dtype_name == "bfloat16":
        array = np.fromfile(path, dtype=np.uint16)
        tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        return tensor.reshape(*shape)

    np_dtype = NUMPY_DTYPE_MAP.get(dtype_name)
    if np_dtype is None:
        raise ValueError(f"Binary load is not implemented for dtype {dtype_name}")
    array = np.fromfile(path, dtype=np_dtype)
    return torch.from_numpy(array.copy()).reshape(*shape)



def save_tensor_txt(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor as text, one value per line.

    This format is useful for studying decimal serialization effects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = tensor.detach().cpu().reshape(-1)
    with path.open("w", encoding="utf-8") as file:
        for value in flat.tolist():
            file.write(f"{value:.17g}\n")



def load_tensor_txt(path: Path, dtype_name: str, shape: Sequence[int]) -> torch.Tensor:
    """Load a tensor from a text file that contains one value per line."""
    with path.open("r", encoding="utf-8") as file:
        values = [line.strip() for line in file if line.strip()]

    torch_dtype = get_torch_dtype(dtype_name)

    if dtype_name in {"float16", "float32", "float64", "bfloat16"}:
        parsed = [float(x) for x in values]
    elif dtype_name in {"int32", "int64"}:
        parsed = [int(x) for x in values]
    else:
        raise ValueError(f"Unsupported dtype for text load: {dtype_name}")

    tensor = torch.tensor(parsed, dtype=torch_dtype)
    return tensor.reshape(*shape)



def save_tensor_any(tensor: torch.Tensor, base_path_without_suffix: Path, formats: Iterable[str], dtype_name: str) -> None:
    """Save a tensor in one or more supported formats."""
    for fmt in formats:
        if fmt == "pt":
            save_tensor_pt(tensor, base_path_without_suffix.with_suffix(".pt"))
        elif fmt == "bin":
            save_tensor_bin(tensor, base_path_without_suffix.with_suffix(".bin"), dtype_name)
        elif fmt == "txt":
            save_tensor_txt(tensor, base_path_without_suffix.with_suffix(".txt"))
        else:
            raise ValueError(f"Unsupported save format: {fmt}")



def load_tensor_any(
    base_path_without_suffix: Path,
    fmt: str,
    dtype_name: str,
    shape: Sequence[int],
    map_location: str = "cpu",
) -> torch.Tensor:
    """Load a tensor from a selected format."""
    if fmt == "pt":
        return load_tensor_pt(base_path_without_suffix.with_suffix(".pt"), map_location=map_location)
    if fmt == "bin":
        return load_tensor_bin(base_path_without_suffix.with_suffix(".bin"), dtype_name, shape)
    if fmt == "txt":
        return load_tensor_txt(base_path_without_suffix.with_suffix(".txt"), dtype_name, shape)
    raise ValueError(f"Unsupported load format: {fmt}")
