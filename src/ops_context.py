"""Execution context used by operation.py.

This class provides:
- input(name): loads one configured input tensor
- op(step_name, function, *args, **kwargs): executes one operation and logs its output
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from .io_utils import save_tensor_any


class OperationContext:
    """Helper object passed to the user-defined operation script."""

    def __init__(
        self,
        inputs: Dict[str, torch.Tensor],
        dtype_name: str,
        save_step_formats: Sequence[str],
        save_final_formats: Sequence[str],
        steps_dir: Path,
        final_dir: Path,
    ) -> None:
        self._inputs = inputs
        self.dtype_name = dtype_name
        self.save_step_formats = list(save_step_formats)
        self.save_final_formats = list(save_final_formats)
        self.steps_dir = steps_dir
        self.final_dir = final_dir
        self.step_index = 0
        self.step_records = []

    def input(self, name: str) -> torch.Tensor:
        """Return one named input tensor."""
        if name not in self._inputs:
            raise KeyError(f"Input tensor '{name}' was not found")
        return self._inputs[name]

    def op(self, step_name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute one operation and log the output if it is a tensor.

        Non-tensor outputs are returned, but only tensor outputs are persisted.
        """
        self.step_index += 1
        result = function(*args, **kwargs)

        if isinstance(result, torch.Tensor):
            file_stem = f"{self.step_index:04d}_{step_name}"
            save_tensor_any(
                tensor=result,
                base_path_without_suffix=self.steps_dir / file_stem,
                formats=self.save_step_formats,
                dtype_name=self.dtype_name,
            )
            self.step_records.append(
                {
                    "step_index": self.step_index,
                    "step_name": step_name,
                    "shape": list(result.shape),
                    "dtype": str(result.dtype),
                }
            )
        else:
            self.step_records.append(
                {
                    "step_index": self.step_index,
                    "step_name": step_name,
                    "shape": None,
                    "dtype": str(type(result)),
                }
            )

        return result

    def save_final_outputs(self, outputs: Mapping[str, Any]) -> None:
        """Save final outputs from the user-defined run function.

        Only tensor values are saved.
        """
        for name, value in outputs.items():
            if isinstance(value, torch.Tensor):
                save_tensor_any(
                    tensor=value,
                    base_path_without_suffix=self.final_dir / name,
                    formats=self.save_final_formats,
                    dtype_name=self.dtype_name,
                )
