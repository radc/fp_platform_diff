"""Comparison utilities for tensors."""

from __future__ import annotations

from typing import Any, Dict

import torch



def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor, rtol: float, atol: float) -> Dict[str, Any]:
    """Compare two tensors and return detailed metrics."""
    if reference.shape != candidate.shape:
        return {
            "same_shape": False,
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
            "allclose": False,
            "exact_equal": False,
        }

    ref = reference.detach().cpu()
    cand = candidate.detach().cpu()

    exact_equal = torch.equal(ref, cand)
    allclose = torch.allclose(ref, cand, rtol=rtol, atol=atol)

    abs_diff = torch.abs(ref - cand)
    max_abs_diff = float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0
    mean_abs_diff = float(abs_diff.mean().item()) if abs_diff.numel() > 0 else 0.0

    denom = torch.abs(ref).clamp_min(1e-30)
    rel_diff = abs_diff / denom
    max_rel_diff = float(rel_diff.max().item()) if rel_diff.numel() > 0 else 0.0
    mean_rel_diff = float(rel_diff.mean().item()) if rel_diff.numel() > 0 else 0.0

    unequal_mask = ref != cand
    num_unequal = int(unequal_mask.sum().item())

    first_diff_index = None
    if num_unequal > 0:
        indices = torch.nonzero(unequal_mask, as_tuple=False)
        first_diff_index = indices[0].tolist()

    return {
        "same_shape": True,
        "shape_reference": list(ref.shape),
        "shape_candidate": list(cand.shape),
        "exact_equal": exact_equal,
        "allclose": allclose,
        "num_unequal": num_unequal,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "first_diff_index": first_diff_index,
    }
