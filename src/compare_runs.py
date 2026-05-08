"""Comparison stage: compare one reference run against one or more candidate runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .comparators import compare_tensors
from .io_utils import load_tensor_any
from .metadata import save_json


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_step_map(step_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a mapping from the saved file stem to step metadata."""
    output: Dict[str, Dict[str, Any]] = {}
    for record in step_records:
        file_stem = f"{record['step_index']:04d}_{record['step_name']}"
        output[file_stem] = record
    return output


def _step_detail(record: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Return step metadata fields that help identify comparison errors."""
    return {
        f"{prefix}_step_index": record.get("step_index"),
        f"{prefix}_step_name": record.get("step_name"),
        f"{prefix}_function_name": record.get("function_name"),
        f"{prefix}_shape": record.get("shape"),
        f"{prefix}_dtype": record.get("dtype"),
    }


def _build_step_error_result(
    *,
    step_name: str,
    ref_record: Dict[str, Any],
    cand_record: Dict[str, Any],
    error: Exception,
) -> Dict[str, Any]:
    """Create a report item for a step that could not be loaded or compared."""
    result: Dict[str, Any] = {
        "step_name": step_name,
        "status": "comparison_error",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    result.update(_step_detail(ref_record, "reference"))
    result.update(_step_detail(cand_record, "candidate"))
    return result


def _pick_worst_step(
    comparisons: List[Dict[str, Any]],
    metric_key: str,
) -> Optional[Dict[str, Any]]:
    """Pick the divergent step with the highest value for a given metric."""
    eligible = [
        item
        for item in comparisons
        if item.get("same_shape") is True
        and item.get("status") is None
        and item.get("exact_equal") is False
        and item.get(metric_key) is not None
    ]

    if not eligible:
        return None

    worst = max(eligible, key=lambda item: item[metric_key])
    return {
        "step_name": worst["step_name"],
        metric_key: worst[metric_key],
    }


def _build_summary(
    comparisons: List[Dict[str, Any]],
    missing_in_candidate: List[str],
    missing_in_reference: List[str],
    first_divergent_step: Optional[str],
) -> Dict[str, Any]:
    """Build a compact summary for a comparison report."""
    skipped_non_tensor = [
        item["step_name"]
        for item in comparisons
        if item.get("status") == "non_tensor_output_skipped"
    ]

    comparison_error_steps = [
        {
            "step_name": item.get("step_name"),
            "reference_function_name": item.get("reference_function_name"),
            "candidate_function_name": item.get("candidate_function_name"),
            "error_type": item.get("error_type"),
            "error_message": item.get("error_message"),
        }
        for item in comparisons
        if item.get("status") == "comparison_error"
    ]

    tensor_steps = [
        item
        for item in comparisons
        if item.get("status") is None and item.get("same_shape") is True
    ]

    shape_mismatch_steps = [
        item["step_name"]
        for item in comparisons
        if item.get("status") is None and item.get("same_shape") is False
    ]

    divergent_tensor_steps = [
        item for item in tensor_steps if item.get("exact_equal") is False
    ]

    total_elements_compared = int(
        sum(item.get("num_elements", 0) or 0 for item in tensor_steps)
    )
    total_unequal_elements = int(
        sum(item.get("num_unequal", 0) or 0 for item in divergent_tensor_steps)
    )
    unequal_element_rate = (
        float(total_unequal_elements / total_elements_compared)
        if total_elements_compared > 0
        else 0.0
    )

    summary = {
        "shared_step_count": len(comparisons),
        "tensor_step_count": len(tensor_steps),
        "exact_equal_tensor_step_count": sum(
            1 for item in tensor_steps if item.get("exact_equal") is True
        ),
        "divergent_tensor_step_count": len(divergent_tensor_steps),
        "shape_mismatch_step_count": len(shape_mismatch_steps),
        "comparison_error_step_count": len(comparison_error_steps),
        "skipped_non_tensor_step_count": len(skipped_non_tensor),
        "missing_in_candidate_count": len(missing_in_candidate),
        "missing_in_reference_count": len(missing_in_reference),
        "first_divergent_step": first_divergent_step,
        "total_elements_compared": total_elements_compared,
        "total_unequal_elements": total_unequal_elements,
        "unequal_element_rate": unequal_element_rate,
        "worst_step_by_max_abs_diff": _pick_worst_step(
            comparisons, "max_abs_diff"
        ),
        "worst_step_by_max_rel_diff": _pick_worst_step(
            comparisons, "max_rel_diff"
        ),
        "worst_step_by_num_unequal": _pick_worst_step(
            comparisons, "num_unequal"
        ),
        "shape_mismatch_steps": shape_mismatch_steps,
        "comparison_error_steps": comparison_error_steps,
        "skipped_non_tensor_steps": skipped_non_tensor,
    }

    return summary


def _build_errors_only_report(full_report: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact report containing only actual mismatches and error statistics."""
    divergent_steps = []
    for item in full_report["comparisons"]:
        if item.get("status") == "non_tensor_output_skipped":
            continue

        if item.get("status") == "comparison_error":
            divergent_steps.append(item)
            continue

        if item.get("same_shape") is False:
            divergent_steps.append(item)
            continue

        if item.get("exact_equal") is False:
            divergent_steps.append(item)

    return {
        "reference_dir": full_report["reference_dir"],
        "candidate_dir": full_report["candidate_dir"],
        "tensor_format": full_report["tensor_format"],
        "rtol": full_report["rtol"],
        "atol": full_report["atol"],
        "first_divergent_step": full_report["first_divergent_step"],
        "missing_in_candidate": full_report["missing_in_candidate"],
        "missing_in_reference": full_report["missing_in_reference"],
        "summary": full_report["summary"],
        "divergent_steps": divergent_steps,
    }


def _compare_single_candidate(
    reference_dir: Path,
    candidate_dir: Path,
    tensor_format: str,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Compare one reference execution folder against one candidate execution folder."""
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

    comparisons: List[Dict[str, Any]] = []
    first_divergent_step: Optional[str] = None

    for step_name in shared_step_names:
        ref_record = ref_steps[step_name]
        cand_record = cand_steps[step_name]
        ref_shape = ref_record["shape"]
        cand_shape = cand_record["shape"]

        if ref_shape is None or cand_shape is None:
            result = {
                "step_name": step_name,
                "status": "non_tensor_output_skipped",
            }
            result.update(_step_detail(ref_record, "reference"))
            result.update(_step_detail(cand_record, "candidate"))
            comparisons.append(result)
            continue

        try:
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
            result["status"] = None
            result.update(_step_detail(ref_record, "reference"))
            result.update(_step_detail(cand_record, "candidate"))
        except Exception as error:
            result = _build_step_error_result(
                step_name=step_name,
                ref_record=ref_record,
                cand_record=cand_record,
                error=error,
            )

        comparisons.append(result)

        if first_divergent_step is None and (
            result.get("exact_equal") is False
            or result.get("status") == "comparison_error"
        ):
            first_divergent_step = step_name

    summary = _build_summary(
        comparisons=comparisons,
        missing_in_candidate=missing_in_candidate,
        missing_in_reference=missing_in_reference,
        first_divergent_step=first_divergent_step,
    )

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
        "summary": summary,
        "comparisons": comparisons,
    }

    return report


def compare_runs(
    reference_dir: Path,
    candidate_dirs: List[Path],
    tensor_format: str,
    rtol: float,
    atol: float,
) -> None:
    """Compare one reference run against one or more candidate runs.

    For each candidate, two JSON files are saved:
    - a full comparison report
    - an errors-only report

    An aggregate summary JSON is also saved in the reference directory.
    """
    aggregate_summary: Dict[str, Any] = {
        "reference_dir": str(reference_dir),
        "tensor_format": tensor_format,
        "rtol": rtol,
        "atol": atol,
        "candidates": [],
    }

    for candidate_dir in candidate_dirs:
        print(f"Comparing candidate: {candidate_dir}")

        full_report_path = (
            candidate_dir
            / f"comparison_against_{reference_dir.name}_{tensor_format}.json"
        )
        errors_only_path = (
            candidate_dir
            / f"comparison_errors_against_{reference_dir.name}_{tensor_format}.json"
        )

        try:
            report = _compare_single_candidate(
                reference_dir=reference_dir,
                candidate_dir=candidate_dir,
                tensor_format=tensor_format,
                rtol=rtol,
                atol=atol,
            )
            errors_only_report = _build_errors_only_report(report)

            save_json(report, full_report_path)
            save_json(errors_only_report, errors_only_path)

            print(f"Full comparison report saved to: {full_report_path}")
            print(f"Errors-only report saved to: {errors_only_path}")

            first_divergent_step = report["first_divergent_step"]
            comparison_error_count = report["summary"].get("comparison_error_step_count", 0)
            if first_divergent_step is None:
                print("No divergence detected in the shared tensor steps.")
            else:
                print(f"First divergent step: {first_divergent_step}")
            if comparison_error_count:
                print(f"Step comparison errors captured: {comparison_error_count}")

            aggregate_summary["candidates"].append(
                {
                    "candidate_dir": str(candidate_dir),
                    "status": "ok",
                    "first_divergent_step": report["first_divergent_step"],
                    "summary": report["summary"],
                    "full_report_path": str(full_report_path),
                    "errors_only_report_path": str(errors_only_path),
                }
            )

        except Exception as error:
            error_report = {
                "reference_dir": str(reference_dir),
                "candidate_dir": str(candidate_dir),
                "tensor_format": tensor_format,
                "rtol": rtol,
                "atol": atol,
                "status": "failed",
                "error_message": str(error),
            }
            save_json(error_report, errors_only_path)

            print(f"Comparison failed for candidate: {candidate_dir}")
            print(f"Failure report saved to: {errors_only_path}")
            print(f"Reason: {error}")

            aggregate_summary["candidates"].append(
                {
                    "candidate_dir": str(candidate_dir),
                    "status": "failed",
                    "error_message": str(error),
                    "errors_only_report_path": str(errors_only_path),
                }
            )

    aggregate_path = (
        reference_dir / f"comparison_batch_summary_{tensor_format}.json"
    )
    save_json(aggregate_summary, aggregate_path)
    print(f"Aggregate comparison summary saved to: {aggregate_path}")