"""Comparison utilities for tensors."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


def _count_special_values(tensor: torch.Tensor) -> Dict[str, int]:
    """Count NaN and Inf values for floating-point tensors."""
    if tensor.is_floating_point():
        return {
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "inf_count": int(torch.isinf(tensor).sum().item()),
        }

    return {
        "nan_count": 0,
        "inf_count": 0,
    }


def _safe_scalar_at_index(tensor: torch.Tensor, index: Optional[List[int]]) -> Any:
    """Return a Python scalar at a given tensor index, or None if unavailable."""
    if index is None:
        return None

    value = tensor[tuple(index)]
    if value.numel() != 1:
        return None

    return value.item()


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Compare two tensors and return detailed metrics.

    The returned dictionary is JSON-serializable and intended to be saved in reports.
    """
    if reference.shape != candidate.shape:
        return {
            "same_shape": False,
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
            "exact_equal": False,
            "allclose": False,
            "num_elements": None,
            "num_unequal": None,
            "fraction_unequal": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rmse_abs_diff": None,
            "max_rel_diff": None,
            "mean_rel_diff": None,
            "first_diff_index": None,
            "first_diff_reference_value": None,
            "first_diff_candidate_value": None,
        }

    ref = reference.detach().cpu()
    cand = candidate.detach().cpu()

    exact_equal = torch.equal(ref, cand)
    allclose = torch.allclose(ref, cand, rtol=rtol, atol=atol)

    num_elements = int(ref.numel())

    # Convert to float64 for robust numeric error statistics.
    ref64 = ref.to(torch.float64)
    cand64 = cand.to(torch.float64)

    abs_diff = torch.abs(ref64 - cand64)
    squared_diff = abs_diff * abs_diff

    max_abs_diff = float(abs_diff.max().item()) if num_elements > 0 else 0.0
    mean_abs_diff = float(abs_diff.mean().item()) if num_elements > 0 else 0.0
    rmse_abs_diff = (
        float(math.sqrt(squared_diff.mean().item())) if num_elements > 0 else 0.0
    )

    denom = torch.abs(ref64).clamp_min(1e-30)
    rel_diff = abs_diff / denom

    max_rel_diff = float(rel_diff.max().item()) if num_elements > 0 else 0.0
    mean_rel_diff = float(rel_diff.mean().item()) if num_elements > 0 else 0.0

    unequal_mask = ref != cand
    num_unequal = int(unequal_mask.sum().item())
    fraction_unequal = float(num_unequal / num_elements) if num_elements > 0 else 0.0

    first_diff_index = None
    if num_unequal > 0:
        indices = torch.nonzero(unequal_mask, as_tuple=False)
        first_diff_index = indices[0].tolist()

    special_ref = _count_special_values(ref)
    special_cand = _count_special_values(cand)

    return {
        "same_shape": True,
        "shape_reference": list(ref.shape),
        "shape_candidate": list(cand.shape),
        "exact_equal": exact_equal,
        "allclose": allclose,
        "num_elements": num_elements,
        "num_unequal": num_unequal,
        "fraction_unequal": fraction_unequal,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rmse_abs_diff": rmse_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "first_diff_index": first_diff_index,
        "first_diff_reference_value": _safe_scalar_at_index(ref, first_diff_index),
        "first_diff_candidate_value": _safe_scalar_at_index(cand, first_diff_index),
        "reference_nan_count": special_ref["nan_count"],
        "reference_inf_count": special_ref["inf_count"],
        "candidate_nan_count": special_cand["nan_count"],
        "candidate_inf_count": special_cand["inf_count"],
    }