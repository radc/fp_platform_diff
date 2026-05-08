#!/usr/bin/env python3
"""Analyze full and error-only cross-platform comparison reports.

This script extends the previous error-oriented analyzer to support two kinds of
JSON reports produced by the floating-point comparison pipeline:

1. Full comparison reports, which contain the key ``comparisons`` and therefore
   include *all* evaluated operations, including those that match exactly.
2. Error-only comparison reports, which contain the key ``divergent_steps`` and
   therefore include only the operations that diverged.

The script can analyze both report families in a single run and writes separate
CSV/PNG outputs for:
- all-step analysis (based on full reports), and
- error-only analysis (based on error reports).

It also generates NxN platform matrices, summaries by operation/family/group,
patch-size analyses, and symmetry analyses comparing A->B against B->A.

The categorization logic mirrors the current operation.py shared in the
conversation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Argument parsing and utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze full and error-only NxN comparison reports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing comparison report JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSVs and plots will be written.",
    )
    parser.add_argument(
        "--full-glob",
        type=str,
        default="**/comparison_against_*.json",
        help="Recursive glob used to find full comparison reports.",
    )
    parser.add_argument(
        "--errors-glob",
        type=str,
        default="**/comparison_errors_against_*.json",
        help="Recursive glob used to find error-only comparison reports.",
    )
    parser.add_argument(
        "--top-k-operations",
        type=int,
        default=25,
        help="Number of top operations to use in bar plots and compact tables.",
    )
    parser.add_argument(
        "--max-annotation-len",
        type=int,
        default=18,
        help="Maximum annotation length used in text heatmaps.",
    )
    parser.add_argument(
        "--name-map",
        action="append",
        default=[],
        help=(
            "Optional execution-name mapping in the form RAW=FRIENDLY_LABEL. "
            "May be passed multiple times. Example: PCEuler_gpu='Platform A:GPU'"
        ),
    )
    parser.add_argument(
        "--name-map-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a dictionary that maps raw execution "
            "folder names to friendly labels."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def basename_from_run_dir(path_str: str) -> str:
    return Path(path_str).name


def shorten(text: Optional[str], max_len: int) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def parse_name_map_entries(entries: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --name-map entry '{entry}'. Expected RAW=FRIENDLY_LABEL."
            )
        raw, friendly = entry.split("=", 1)
        raw = raw.strip()
        friendly = friendly.strip()
        if not raw or not friendly:
            raise ValueError(
                f"Invalid --name-map entry '{entry}'. RAW and FRIENDLY_LABEL must be non-empty."
            )
        mapping[raw] = friendly
    return mapping


def load_name_map(args: argparse.Namespace) -> Dict[str, str]:
    mapping = parse_name_map_entries(args.name_map)
    if args.name_map_json is not None:
        with args.name_map_json.open("r", encoding="utf-8") as f:
            file_mapping = json.load(f)
        if not isinstance(file_mapping, dict):
            raise ValueError("--name-map-json must contain a JSON object/dictionary.")
        mapping.update({str(k): str(v) for k, v in file_mapping.items()})
    return mapping


def apply_name_map(name: str, name_map: Dict[str, str]) -> str:
    return name_map.get(name, name)


# -----------------------------------------------------------------------------
# Step parsing and categorization
# -----------------------------------------------------------------------------


def parse_step_name(step_name: str) -> Tuple[Optional[int], str]:
    """Return ``(step_index, normalized_name_without_numeric_prefix)``."""
    match = re.match(r"^(\d+)_(.+)$", step_name)
    if match:
        return int(match.group(1)), match.group(2)
    return None, step_name


def extract_patch_size(normalized_name: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", normalized_name)
    if match and "patch" in normalized_name:
        return int(match.group(1))
    return None


def precision_variant(normalized_name: str) -> str:
    return "fp64" if "_fp64" in normalized_name else "fp32"


def canonical_operation_name(normalized_name: str) -> str:
    """Keep the operation name informative for the paper and analysis.

    The goal is to preserve distinctions such as:
    - cumsum_a vs cumsum_b
    - patch_cumsum_a_9 vs patch_cumsum_a_25
    - patch_sum_mul_ab_9 vs patch_sum_mul_cd_25
    """
    return normalized_name


def operation_family(normalized_name: str) -> str:
    """Classify a step according to the current operation.py design."""
    if normalized_name.startswith(("left_assoc_", "right_assoc_", "tree_assoc_", "assoc_diff_")):
        return "associativity_tests"

    if normalized_name.startswith("repeated_add_"):
        return "control_repeated_add"

    if "_trim_patch_" in normalized_name or "_reshape_patch_" in normalized_name:
        return "patch_preparation"

    if normalized_name.startswith("patch_"):
        if normalized_name.startswith("patch_cumsum"):
            return "patch_cumulative"
        if normalized_name.startswith((
            "patch_sum",
            "patch_mean",
            "patch_std",
            "patch_var",
            "patch_norm",
            "patch_amax",
            "patch_amin",
        )):
            return "patch_reduction"
        if normalized_name.startswith("patch_softmax"):
            return "patch_softmax"
        return "patch_other"

    if normalized_name.startswith(("sum_", "mean_", "std_", "var_", "norm_", "max_", "min_")):
        return "global_reduction"

    if normalized_name.startswith("cumsum_"):
        return "global_cumulative"

    if normalized_name.startswith(("addcmul_", "addcdiv_", "lerp_")):
        return "mixed_fused"

    if normalized_name.startswith((
        "sqrt_",
        "rsqrt_",
        "log_",
        "log1p_",
        "exp_",
        "expm1_",
        "sin_",
        "cos_",
        "tan_",
        "arcsin_",
        "arccos_",
        "arctan_",
        "atan2_",
        "sinh_",
        "cosh_",
        "tanh_",
        "erf_",
        "sigmoid_",
        "relu_",
        "leaky_relu_",
        "softplus_",
        "hypot_",
    )):
        return "transcendental_nonlinear"

    if normalized_name.startswith((
        "abs_",
        "neg_",
        "sign_",
        "round_",
        "floor_",
        "ceil_",
        "trunc_",
        "frac_",
        "clamp_",
    )):
        return "rounding_sign"

    if normalized_name.startswith(("square_", "cube_", "pow_")):
        return "power_like"

    if normalized_name.startswith(("div_", "reciprocal_")):
        return "division_reciprocal"

    if normalized_name.startswith(("add_", "sub_", "mul_")):
        return "elementwise_arithmetic"

    return "other"


def operation_group_for_paper(normalized_name: str) -> str:
    """Higher-level grouping suitable for paper tables and figures."""
    family = operation_family(normalized_name)
    if family in {"elementwise_arithmetic", "division_reciprocal", "power_like"}:
        return "elementwise_arithmetic"
    if family == "associativity_tests":
        return "associativity_tests"
    if family == "rounding_sign":
        return "rounding_sign"
    if family == "transcendental_nonlinear":
        return "transcendental_nonlinear"
    if family == "mixed_fused":
        return "mixed_fused"
    if family in {"global_reduction", "global_cumulative"}:
        return "global_reduction_cumulative"
    if family == "control_repeated_add":
        return "control_repeated_add"
    if family in {"patch_preparation", "patch_reduction", "patch_cumulative", "patch_softmax", "patch_other"}:
        return "patch_based"
    return "other"


# -----------------------------------------------------------------------------
# Report loading
# -----------------------------------------------------------------------------


def detect_report_kind(report: Dict[str, Any]) -> str:
    if "comparisons" in report:
        return "full"
    if "divergent_steps" in report:
        return "errors"
    return "unknown"


def iter_report_items(report: Dict[str, Any], report_kind: str) -> Iterable[Dict[str, Any]]:
    if report_kind == "full":
        return report.get("comparisons", [])
    if report_kind == "errors":
        return report.get("divergent_steps", [])
    return []


def step_row_from_item(
    *,
    report_path: Path,
    report_kind: str,
    reference_name: str,
    candidate_name: str,
    reference_raw: Optional[str],
    candidate_raw: Optional[str],
    item: Dict[str, Any],
) -> Dict[str, Any]:
    step_name = item["step_name"]
    step_index, normalized_name = parse_step_name(step_name)

    exact_equal = bool(item.get("exact_equal", False))
    allclose = bool(item.get("allclose", False))
    same_shape = bool(item.get("same_shape", False))
    num_unequal = int(item.get("num_unequal", 0) or 0)
    num_elements = int(item.get("num_elements", 0) or 0)
    fraction_unequal = float(item.get("fraction_unequal", 0.0) or 0.0)

    divergent = (not exact_equal) or (num_unequal > 0)

    return {
        "report_path": str(report_path),
        "report_kind": report_kind,
        "reference": reference_name,
        "candidate": candidate_name,
        "reference_raw": reference_raw,
        "candidate_raw": candidate_raw,
        "step_name": step_name,
        "step_index": step_index,
        "operation": canonical_operation_name(normalized_name),
        "family": operation_family(normalized_name),
        "group": operation_group_for_paper(normalized_name),
        "patch_size": extract_patch_size(normalized_name),
        "precision_variant": precision_variant(normalized_name),
        "exact_equal": exact_equal,
        "allclose": allclose,
        "same_shape": same_shape,
        "divergent": divergent,
        "num_elements": num_elements,
        "num_unequal": num_unequal,
        "fraction_unequal": fraction_unequal,
        "max_abs_diff": float(item.get("max_abs_diff", 0.0) or 0.0),
        "mean_abs_diff": float(item.get("mean_abs_diff", 0.0) or 0.0),
        "rmse_abs_diff": float(item.get("rmse_abs_diff", 0.0) or 0.0),
        "max_rel_diff": float(item.get("max_rel_diff", 0.0) or 0.0),
        "mean_rel_diff": float(item.get("mean_rel_diff", 0.0) or 0.0),
        "candidate_nan_count": int(item.get("candidate_nan_count", 0) or 0),
        "candidate_inf_count": int(item.get("candidate_inf_count", 0) or 0),
        "reference_nan_count": int(item.get("reference_nan_count", 0) or 0),
        "reference_inf_count": int(item.get("reference_inf_count", 0) or 0),
        "first_diff_index": json.dumps(item.get("first_diff_index")),
        "first_diff_reference_value": item.get("first_diff_reference_value"),
        "first_diff_candidate_value": item.get("first_diff_candidate_value"),
        "status": item.get("status"),
    }


def load_report_family(paths: Sequence[Path], name_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load one family of reports and return pair-level and step-level frames."""
    pair_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for path in sorted(paths):
        try:
            report = load_json(path)
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"Could not read {path}: {exc}")
            continue

        report_kind = detect_report_kind(report)
        if report_kind == "unknown":
            warnings.append(f"Skipping {path}: unrecognized JSON structure.")
            continue

        reference_raw = basename_from_run_dir(report.get("reference_dir", "unknown_reference"))
        candidate_raw = basename_from_run_dir(report.get("candidate_dir", "unknown_candidate"))
        reference_name = apply_name_map(reference_raw, name_map)
        candidate_name = apply_name_map(candidate_raw, name_map)

        if report.get("status") == "failed":
            pair_rows.append(
                {
                    "report_path": str(path),
                    "report_kind": report_kind,
                    "reference": reference_name,
                    "candidate": candidate_name,
                    "reference_raw": reference_raw,
                    "candidate_raw": candidate_raw,
                    "status": "failed",
                    "error_message": report.get("error_message"),
                }
            )
            continue

        summary = report.get("summary", {}) or {}
        first_divergent_step = report.get("first_divergent_step")
        _, first_divergent_normalized = parse_step_name(first_divergent_step) if first_divergent_step else (None, None)

        pair_rows.append(
            {
                "report_path": str(path),
                "report_kind": report_kind,
                "reference": reference_name,
                "candidate": candidate_name,
                "status": "ok",
                "tensor_format": report.get("tensor_format"),
                "rtol": report.get("rtol"),
                "atol": report.get("atol"),
                "first_divergent_step": first_divergent_step,
                "first_divergent_operation": first_divergent_normalized,
                "first_divergent_family": operation_family(first_divergent_normalized) if first_divergent_normalized else None,
                "first_divergent_group": operation_group_for_paper(first_divergent_normalized) if first_divergent_normalized else None,
                "shared_step_count": summary.get("shared_step_count"),
                "tensor_step_count": summary.get("tensor_step_count"),
                "exact_equal_tensor_step_count": summary.get("exact_equal_tensor_step_count"),
                "divergent_tensor_step_count": summary.get("divergent_tensor_step_count"),
                "shape_mismatch_step_count": summary.get("shape_mismatch_step_count"),
                "missing_in_candidate_count": summary.get("missing_in_candidate_count"),
                "missing_in_reference_count": summary.get("missing_in_reference_count"),
                "total_elements_compared": summary.get("total_elements_compared"),
                "total_unequal_elements": summary.get("total_unequal_elements"),
                "unequal_element_rate": summary.get("unequal_element_rate"),
                "worst_step_by_max_abs_diff": (summary.get("worst_step_by_max_abs_diff") or {}).get("step_name"),
                "worst_step_by_max_abs_diff_value": (summary.get("worst_step_by_max_abs_diff") or {}).get("max_abs_diff"),
                "worst_step_by_max_rel_diff": (summary.get("worst_step_by_max_rel_diff") or {}).get("step_name"),
                "worst_step_by_max_rel_diff_value": (summary.get("worst_step_by_max_rel_diff") or {}).get("max_rel_diff"),
                "worst_step_by_num_unequal": (summary.get("worst_step_by_num_unequal") or {}).get("step_name"),
                "worst_step_by_num_unequal_value": (summary.get("worst_step_by_num_unequal") or {}).get("num_unequal"),
                "num_items_in_report": len(list(iter_report_items(report, report_kind))),
            }
        )

        for item in iter_report_items(report, report_kind):
            step_rows.append(
                step_row_from_item(
                    report_path=path,
                    report_kind=report_kind,
                    reference_name=reference_name,
                    candidate_name=candidate_name,
                    reference_raw=reference_raw,
                    candidate_raw=candidate_raw,
                    item=item,
                )
            )

    pair_df = pd.DataFrame(pair_rows)
    step_df = pd.DataFrame(step_rows)

    if not pair_df.empty and {"tensor_step_count", "exact_equal_tensor_step_count", "divergent_tensor_step_count"}.issubset(pair_df.columns):
        pair_df["exact_equal_tensor_step_rate"] = pair_df["exact_equal_tensor_step_count"] / pair_df["tensor_step_count"]
        pair_df["divergent_tensor_step_rate"] = pair_df["divergent_tensor_step_count"] / pair_df["tensor_step_count"]

    return pair_df, step_df, warnings


# -----------------------------------------------------------------------------
# Aggregations
# -----------------------------------------------------------------------------


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def aggregate_platform_role(pair_df: pd.DataFrame, role_column: str) -> pd.DataFrame:
    metric_cols = [
        "divergent_tensor_step_count",
        "divergent_tensor_step_rate",
        "exact_equal_tensor_step_count",
        "exact_equal_tensor_step_rate",
        "total_elements_compared",
        "total_unequal_elements",
        "unequal_element_rate",
    ]
    existing_metric_cols = [c for c in metric_cols if c in pair_df.columns]
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(columns=["platform"])
    grouped = ok_df.groupby(role_column)[existing_metric_cols].agg(["mean", "median", "max", "min"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index().rename(columns={role_column: "platform"})
    return grouped.sort_values("platform")


def aggregate_all_steps(step_df: pd.DataFrame) -> pd.DataFrame:
    if step_df.empty:
        return pd.DataFrame()
    grouped = (
        step_df.groupby(["operation", "family", "group", "precision_variant", "patch_size"], dropna=False)
        .agg(
            num_platform_pairs=("report_path", "nunique"),
            observations=("step_name", "count"),
            exact_equal_count=("exact_equal", lambda s: int(pd.Series(s).fillna(False).sum())),
            allclose_count=("allclose", lambda s: int(pd.Series(s).fillna(False).sum())),
            divergent_count=("divergent", lambda s: int(pd.Series(s).fillna(False).sum())),
            same_shape_count=("same_shape", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_fraction_unequal=("fraction_unequal", "mean"),
            median_fraction_unequal=("fraction_unequal", "median"),
            max_fraction_unequal=("fraction_unequal", "max"),
            mean_max_abs_diff=("max_abs_diff", "mean"),
            max_max_abs_diff=("max_abs_diff", "max"),
            mean_max_rel_diff=("max_rel_diff", "mean"),
            max_max_rel_diff=("max_rel_diff", "max"),
            mean_num_unequal=("num_unequal", "mean"),
            max_num_unequal=("num_unequal", "max"),
            total_num_unequal=("num_unequal", "sum"),
            total_num_elements=("num_elements", "sum"),
        )
        .reset_index()
    )

    grouped["exact_equal_rate"] = grouped["exact_equal_count"] / grouped["observations"]
    grouped["allclose_rate"] = grouped["allclose_count"] / grouped["observations"]
    grouped["divergent_rate"] = grouped["divergent_count"] / grouped["observations"]
    grouped["same_shape_rate"] = grouped["same_shape_count"] / grouped["observations"]
    grouped["global_unequal_element_rate"] = grouped["total_num_unequal"] / grouped["total_num_elements"].replace(0, np.nan)

    return grouped.sort_values(
        ["divergent_rate", "mean_fraction_unequal", "max_max_abs_diff"],
        ascending=[False, False, False],
    )


def aggregate_all_steps_by_family(step_df: pd.DataFrame) -> pd.DataFrame:
    if step_df.empty:
        return pd.DataFrame()
    grouped = (
        step_df.groupby(["family", "group", "precision_variant"], dropna=False)
        .agg(
            num_platform_pairs=("report_path", "nunique"),
            observations=("step_name", "count"),
            unique_operations=("operation", "nunique"),
            exact_equal_count=("exact_equal", lambda s: int(pd.Series(s).fillna(False).sum())),
            allclose_count=("allclose", lambda s: int(pd.Series(s).fillna(False).sum())),
            divergent_count=("divergent", lambda s: int(pd.Series(s).fillna(False).sum())),
            same_shape_count=("same_shape", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_fraction_unequal=("fraction_unequal", "mean"),
            median_fraction_unequal=("fraction_unequal", "median"),
            max_fraction_unequal=("fraction_unequal", "max"),
            mean_max_abs_diff=("max_abs_diff", "mean"),
            max_max_abs_diff=("max_abs_diff", "max"),
            mean_max_rel_diff=("max_rel_diff", "mean"),
            max_max_rel_diff=("max_rel_diff", "max"),
            mean_num_unequal=("num_unequal", "mean"),
            max_num_unequal=("num_unequal", "max"),
            total_num_unequal=("num_unequal", "sum"),
            total_num_elements=("num_elements", "sum"),
        )
        .reset_index()
    )

    grouped["exact_equal_rate"] = grouped["exact_equal_count"] / grouped["observations"]
    grouped["allclose_rate"] = grouped["allclose_count"] / grouped["observations"]
    grouped["divergent_rate"] = grouped["divergent_count"] / grouped["observations"]
    grouped["same_shape_rate"] = grouped["same_shape_count"] / grouped["observations"]
    grouped["global_unequal_element_rate"] = grouped["total_num_unequal"] / grouped["total_num_elements"].replace(0, np.nan)

    return grouped.sort_values(["divergent_rate", "mean_fraction_unequal"], ascending=[False, False])


def aggregate_patch_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    patch_df = step_df[step_df["patch_size"].notna()].copy()
    if patch_df.empty:
        return pd.DataFrame()
    grouped = (
        patch_df.groupby(["patch_size", "family", "operation", "precision_variant"], dropna=False)
        .agg(
            observations=("step_name", "count"),
            exact_equal_count=("exact_equal", lambda s: int(pd.Series(s).fillna(False).sum())),
            divergent_count=("divergent", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_fraction_unequal=("fraction_unequal", "mean"),
            max_fraction_unequal=("fraction_unequal", "max"),
            mean_max_abs_diff=("max_abs_diff", "mean"),
            max_max_abs_diff=("max_abs_diff", "max"),
            mean_max_rel_diff=("max_rel_diff", "mean"),
            max_max_rel_diff=("max_rel_diff", "max"),
            total_num_unequal=("num_unequal", "sum"),
            total_num_elements=("num_elements", "sum"),
        )
        .reset_index()
    )
    grouped["exact_equal_rate"] = grouped["exact_equal_count"] / grouped["observations"]
    grouped["divergent_rate"] = grouped["divergent_count"] / grouped["observations"]
    grouped["global_unequal_element_rate"] = grouped["total_num_unequal"] / grouped["total_num_elements"].replace(0, np.nan)
    return grouped.sort_values(["patch_size", "divergent_rate"], ascending=[True, False])


def summarize_first_divergent(pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_operation = ok_df.groupby("first_divergent_operation", dropna=False).size().reset_index(name="count")
    by_operation = by_operation.sort_values("count", ascending=False)
    by_family = ok_df.groupby("first_divergent_family", dropna=False).size().reset_index(name="count")
    by_family = by_family.sort_values("count", ascending=False)
    return by_operation, by_family


def build_matrix(pair_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame()
    matrix = ok_df.pivot(index="reference", columns="candidate", values=value_col)
    platforms = sorted(set(ok_df["reference"]).union(set(ok_df["candidate"])))
    matrix = matrix.reindex(index=platforms, columns=platforms)
    for platform in platforms:
        if platform in matrix.index and platform in matrix.columns:
            matrix.loc[platform, platform] = np.nan
    return matrix


def build_text_matrix(pair_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame()
    matrix = ok_df.pivot(index="reference", columns="candidate", values=value_col)
    platforms = sorted(set(ok_df["reference"]).union(set(ok_df["candidate"])))
    matrix = matrix.reindex(index=platforms, columns=platforms)
    for platform in platforms:
        if platform in matrix.index and platform in matrix.columns:
            matrix.loc[platform, platform] = "-"
    return matrix


def symmetry_analysis(pair_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame()

    pair_map = {(row.reference, row.candidate): row for row in ok_df.itertuples(index=False)}
    rows: List[Dict[str, Any]] = []
    platforms = sorted(set(ok_df["reference"]).union(set(ok_df["candidate"])))

    for i, a in enumerate(platforms):
        for b in platforms[i + 1 :]:
            ab = pair_map.get((a, b))
            ba = pair_map.get((b, a))
            if ab is None or ba is None:
                continue
            rows.append(
                {
                    "platform_a": a,
                    "platform_b": b,
                    "a_to_b_unequal_rate": ab.unequal_element_rate,
                    "b_to_a_unequal_rate": ba.unequal_element_rate,
                    "abs_diff_unequal_rate": abs((ab.unequal_element_rate or 0.0) - (ba.unequal_element_rate or 0.0)),
                    "a_to_b_divergent_steps": ab.divergent_tensor_step_count,
                    "b_to_a_divergent_steps": ba.divergent_tensor_step_count,
                    "abs_diff_divergent_steps": abs((ab.divergent_tensor_step_count or 0) - (ba.divergent_tensor_step_count or 0)),
                    "a_to_b_first_divergent_operation": ab.first_divergent_operation,
                    "b_to_a_first_divergent_operation": ba.first_divergent_operation,
                    "same_first_divergent_operation": ab.first_divergent_operation == ba.first_divergent_operation,
                    "a_to_b_first_divergent_family": ab.first_divergent_family,
                    "b_to_a_first_divergent_family": ba.first_divergent_family,
                    "same_first_divergent_family": ab.first_divergent_family == ba.first_divergent_family,
                }
            )

    sym_df = pd.DataFrame(rows)
    if sym_df.empty:
        return sym_df
    return sym_df.sort_values(
        ["abs_diff_unequal_rate", "abs_diff_divergent_steps"],
        ascending=[False, False],
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_numeric_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    value_format: str = ".3g",
) -> None:
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(matrix.columns)), 0.9 * max(5, len(matrix.index))))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = values[i, j]
            text = "-" if math.isnan(val) else format(val, value_format)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_text_heatmap(
    matrix_values: pd.DataFrame,
    matrix_colors: pd.DataFrame,
    title: str,
    output_path: Path,
    max_annotation_len: int,
    cmap: str = "magma",
) -> None:
    if matrix_values.empty or matrix_colors.empty:
        return

    color_values = matrix_colors.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(color_values)

    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(matrix_values.columns)), 0.9 * max(5, len(matrix_values.index))))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(range(len(matrix_values.columns)))
    ax.set_xticklabels(matrix_values.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix_values.index)))
    ax.set_yticklabels(matrix_values.index)

    for i in range(matrix_values.shape[0]):
        for j in range(matrix_values.shape[1]):
            raw_value = matrix_values.iloc[i, j]
            text = shorten(None if pd.isna(raw_value) else str(raw_value), max_annotation_len) or "-"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Color scale", rotation=-90, va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    top_k: int,
) -> None:
    if df.empty or category_col not in df.columns or value_col not in df.columns:
        return

    plot_df = df.dropna(subset=[category_col, value_col]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(value_col, ascending=False).head(top_k)
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df[category_col].astype(str), plot_df[value_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel(category_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Report-family analysis
# -----------------------------------------------------------------------------


def analyze_family(
    *,
    family_name: str,
    pair_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
    top_k_operations: int,
    max_annotation_len: int,
) -> Dict[str, Any]:
    """Analyze one report family and write CSV/PNG outputs."""
    ensure_dir(output_dir)

    outputs: Dict[str, Any] = {
        "family_name": family_name,
        "num_pair_reports": len(pair_df),
        "num_step_rows": len(step_df),
    }

    save_csv(pair_df, output_dir / f"{family_name}_pairwise_summary.csv")
    save_csv(step_df, output_dir / f"{family_name}_steps_long.csv")

    if pair_df.empty:
        return outputs

    role_ref = aggregate_platform_role(pair_df, "reference")
    role_cand = aggregate_platform_role(pair_df, "candidate")
    save_csv(role_ref, output_dir / f"{family_name}_platform_summary_as_reference.csv")
    save_csv(role_cand, output_dir / f"{family_name}_platform_summary_as_candidate.csv")

    if not step_df.empty:
        operation_summary = aggregate_all_steps(step_df)
        family_summary = aggregate_all_steps_by_family(step_df)
        patch_summary = aggregate_patch_summary(step_df)

        save_csv(operation_summary, output_dir / f"{family_name}_operation_summary.csv")
        save_csv(family_summary, output_dir / f"{family_name}_family_summary.csv")
        save_csv(patch_summary, output_dir / f"{family_name}_patch_summary.csv")

        divergent_only = step_df[step_df["divergent"]].copy()
        if not divergent_only.empty:
            error_operation_summary = aggregate_all_steps(divergent_only)
            error_family_summary = aggregate_all_steps_by_family(divergent_only)
            error_patch_summary = aggregate_patch_summary(divergent_only)
            save_csv(error_operation_summary, output_dir / f"{family_name}_divergent_operation_summary.csv")
            save_csv(error_family_summary, output_dir / f"{family_name}_divergent_family_summary.csv")
            save_csv(error_patch_summary, output_dir / f"{family_name}_divergent_patch_summary.csv")

        plot_top_bar(
            operation_summary,
            category_col="operation",
            value_col="divergent_rate",
            title=f"{family_name}: top operations by divergence rate",
            output_path=output_dir / f"{family_name}_top_operations_by_divergence_rate.png",
            top_k=top_k_operations,
        )
        plot_top_bar(
            operation_summary,
            category_col="operation",
            value_col="mean_fraction_unequal",
            title=f"{family_name}: top operations by mean unequal fraction",
            output_path=output_dir / f"{family_name}_top_operations_by_mean_fraction_unequal.png",
            top_k=top_k_operations,
        )
        plot_top_bar(
            family_summary,
            category_col="family",
            value_col="divergent_rate",
            title=f"{family_name}: operation-family divergence rate",
            output_path=output_dir / f"{family_name}_families_by_divergence_rate.png",
            top_k=50,
        )
        plot_top_bar(
            family_summary,
            category_col="family",
            value_col="mean_fraction_unequal",
            title=f"{family_name}: operation-family mean unequal fraction",
            output_path=output_dir / f"{family_name}_families_by_mean_fraction_unequal.png",
            top_k=50,
        )

        if not patch_summary.empty:
            plot_top_bar(
                patch_summary.sort_values("divergent_rate", ascending=False),
                category_col="operation",
                value_col="divergent_rate",
                title=f"{family_name}: patch operations by divergence rate",
                output_path=output_dir / f"{family_name}_patch_operations_by_divergence_rate.png",
                top_k=top_k_operations,
            )

    first_op_df, first_family_df = summarize_first_divergent(pair_df)
    save_csv(first_op_df, output_dir / f"{family_name}_first_divergent_operation_counts.csv")
    save_csv(first_family_df, output_dir / f"{family_name}_first_divergent_family_counts.csv")

    sym_df = symmetry_analysis(pair_df)
    save_csv(sym_df, output_dir / f"{family_name}_symmetry_analysis.csv")

    numeric_matrix_specs = [
        ("unequal_element_rate", ".3g", "viridis"),
        ("divergent_tensor_step_count", ".0f", "magma"),
        ("divergent_tensor_step_rate", ".3g", "magma"),
        ("exact_equal_tensor_step_count", ".0f", "cividis"),
        ("exact_equal_tensor_step_rate", ".3g", "cividis"),
        ("total_unequal_elements", ".3g", "plasma"),
    ]

    for value_col, fmt, cmap in numeric_matrix_specs:
        if value_col not in pair_df.columns:
            continue
        matrix = build_matrix(pair_df, value_col)
        if matrix.empty:
            continue
        matrix.to_csv(output_dir / f"{family_name}_matrix_{value_col}.csv")
        plot_numeric_heatmap(
            matrix,
            title=f"{family_name}: {value_col}",
            output_path=output_dir / f"{family_name}_matrix_{value_col}.png",
            cmap=cmap,
            value_format=fmt,
        )

    text_specs = [
        "first_divergent_operation",
        "first_divergent_family",
        "first_divergent_step",
    ]
    color_reference_matrix = build_matrix(pair_df, "unequal_element_rate") if "unequal_element_rate" in pair_df.columns else pd.DataFrame()

    for value_col in text_specs:
        if value_col not in pair_df.columns or color_reference_matrix.empty:
            continue
        matrix_text = build_text_matrix(pair_df, value_col)
        if matrix_text.empty:
            continue
        matrix_text.to_csv(output_dir / f"{family_name}_matrix_{value_col}.csv")
        plot_text_heatmap(
            matrix_values=matrix_text,
            matrix_colors=color_reference_matrix,
            title=f"{family_name}: {value_col}",
            output_path=output_dir / f"{family_name}_matrix_{value_col}.png",
            max_annotation_len=max_annotation_len,
            cmap="magma",
        )

    outputs["num_ok_pair_reports"] = int((pair_df["status"] == "ok").sum()) if "status" in pair_df.columns else 0
    outputs["num_failed_pair_reports"] = int((pair_df["status"] == "failed").sum()) if "status" in pair_df.columns else 0
    outputs["num_divergent_steps"] = int(step_df["divergent"].sum()) if "divergent" in step_df.columns else 0
    return outputs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    name_map = load_name_map(args)

    full_paths = sorted(set(args.input_dir.glob(args.full_glob)))
    error_paths = sorted(set(args.input_dir.glob(args.errors_glob)))

    # Avoid double-counting when a user provides overlapping globs.
    full_paths = [p for p in full_paths if p not in set(error_paths)]

    inventory_rows: List[Dict[str, Any]] = []
    for path in full_paths:
        inventory_rows.append({"report_path": str(path), "report_kind": "full"})
    for path in error_paths:
        inventory_rows.append({"report_path": str(path), "report_kind": "errors"})
    inventory_df = pd.DataFrame(inventory_rows)
    save_csv(inventory_df, args.output_dir / "reports_inventory.csv")

    full_pair_df, full_step_df, full_warnings = load_report_family(full_paths, name_map)
    error_pair_df, error_step_df, error_warnings = load_report_family(error_paths, name_map)

    summary_lines: List[str] = []
    summary_lines.append("# Comparison report analysis")
    summary_lines.append("")
    summary_lines.append(f"Input directory: `{args.input_dir}`")
    summary_lines.append(f"Full reports found: {len(full_paths)}")
    summary_lines.append(f"Error-only reports found: {len(error_paths)}")
    summary_lines.append("")
    if name_map:
        summary_lines.append("Friendly label mappings:")
        for raw_name, friendly_name in sorted(name_map.items()):
            summary_lines.append(f"- `{raw_name}` -> `{friendly_name}`")
        summary_lines.append("")

    if full_warnings or error_warnings:
        summary_lines.append("## Warnings")
        for msg in full_warnings + error_warnings:
            summary_lines.append(f"- {msg}")
        summary_lines.append("")

    full_outputs = analyze_family(
        family_name="full_reports",
        pair_df=full_pair_df,
        step_df=full_step_df,
        output_dir=args.output_dir / "full_reports",
        top_k_operations=args.top_k_operations,
        max_annotation_len=args.max_annotation_len,
    )

    error_outputs = analyze_family(
        family_name="error_reports",
        pair_df=error_pair_df,
        step_df=error_step_df,
        output_dir=args.output_dir / "error_reports",
        top_k_operations=args.top_k_operations,
        max_annotation_len=args.max_annotation_len,
    )

    summary_lines.append("## Output overview")
    summary_lines.append("")
    for outputs in [full_outputs, error_outputs]:
        summary_lines.append(f"### {outputs['family_name']}")
        summary_lines.append(f"- Pair reports: {outputs.get('num_pair_reports', 0)}")
        summary_lines.append(f"- OK pair reports: {outputs.get('num_ok_pair_reports', 0)}")
        summary_lines.append(f"- Failed pair reports: {outputs.get('num_failed_pair_reports', 0)}")
        summary_lines.append(f"- Step rows: {outputs.get('num_step_rows', 0)}")
        summary_lines.append(f"- Divergent step rows: {outputs.get('num_divergent_steps', 0)}")
        summary_lines.append("")

    (args.output_dir / "analysis_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Analysis completed.")
    print(f"Output directory: {args.output_dir}")
    print(f"Full reports processed: {len(full_paths)}")
    print(f"Error-only reports processed: {len(error_paths)}")


if __name__ == "__main__":
    main()
