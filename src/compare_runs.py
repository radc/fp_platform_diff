"""Comparison stage: compare two execution folders step by step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .comparators import compare_tensors
from .io_utils import load_tensor_any
from .metadata import save_json



def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)



def _build_step_map(step_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a mapping from the saved file stem to step metadata."""
    output = {}
    for record in step_records:
        file_stem = f"{record['step_index']:04d}_{record['step_name']}"
        output[file_stem] = record
    return output



def compare_runs(
    reference_dir: Path,
    candidate_dir: Path,
    tensor_format: str,
    rtol: float,
    atol: float,
) -> None:
    """Compare two execution folders step by step and save a report."""
    ref_meta = _load_json(reference_dir / "metadata" / "execution_metadata.json")
    cand_meta = _load_json(candidate_dir / "metadata" / "execution_metadata.json")

    ref_dtype = ref_meta["dtype"]
    cand_dtype = cand_meta["dtype"]
    if ref_dtype != cand_dtype:
        raise ValueError(f"Dtype mismatch: {ref_dtype} vs {cand_dtype}")

    ref_steps = _build_step_map(ref_meta["step_records"])
    cand_steps = _build_step_map(cand_meta["step_records"])

    shared_step_names = sorted(set(ref_steps.keys()) & set(cand_steps.keys()))
    missing_in_candidate = sorted(set(ref_steps.keys()) - set(cand_steps.keys()))
    missing_in_reference = sorted(set(cand_steps.keys()) - set(ref_steps.keys()))

    comparisons = []
    first_divergent_step = None

    for step_name in shared_step_names:
        ref_shape = ref_steps[step_name]["shape"]
        cand_shape = cand_steps[step_name]["shape"]

        if ref_shape is None or cand_shape is None:
            comparisons.append(
                {
                    "step_name": step_name,
                    "status": "non_tensor_output_skipped",
                }
            )
            continue

        ref_tensor = load_tensor_any(
            base_path_without_suffix=reference_dir / "steps" / step_name,
            fmt=tensor_format,
            dtype_name=ref_dtype,
            shape=ref_shape,
            map_location="cpu",
        )
        cand_tensor = load_tensor_any(
            base_path_without_suffix=candidate_dir / "steps" / step_name,
            fmt=tensor_format,
            dtype_name=cand_dtype,
            shape=cand_shape,
            map_location="cpu",
        )

        result = compare_tensors(ref_tensor, cand_tensor, rtol=rtol, atol=atol)
        result["step_name"] = step_name
        comparisons.append(result)

        if first_divergent_step is None:
            if not result.get("exact_equal", False):
                first_divergent_step = step_name

    report = {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "tensor_format": tensor_format,
        "rtol": rtol,
        "atol": atol,
        "shared_steps": shared_step_names,
        "missing_in_candidate": missing_in_candidate,
        "missing_in_reference": missing_in_reference,
        "first_divergent_step": first_divergent_step,
        "comparisons": comparisons,
    }

    report_path = candidate_dir / f"comparison_against_{reference_dir.name}_{tensor_format}.json"
    save_json(report, report_path)

    print(f"Comparison report saved to: {report_path}")
    if first_divergent_step is None:
        print("No divergence detected in the shared tensor steps.")
    else:
        print(f"First divergent step: {first_divergent_step}")
