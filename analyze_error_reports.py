#!/usr/bin/env python3
"""Analyze cross-platform floating-point error reports.

This script reads the JSON reports produced by the comparison stage
(e.g. comparison_errors_against_<reference>_<format>.json), aggregates the
results across all platform pairs, and generates tables/plots that are useful
for cross-platform numerical analysis and for writing a paper.

Main features
-------------
1. Builds an N x N platform matrix using every report found recursively.
2. Aggregates statistics by pair of platforms, by operation, by family of
   operations, by patch size, and by precision variant (fp32/fp64).
3. Identifies the most divergence-prone operations and families.
4. Compares A->B against B->A to quantify directional asymmetry.
5. Generates CSV files and PNG plots ready to inspect or include in a paper.

The categorization logic mirrors the operation groups in the current
operation.py shared in the conversation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze NxN cross-platform floating-point error reports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory that contains comparison error JSON files. The search is recursive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV files and plots will be written.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/comparison_errors_against_*.json",
        help="Recursive glob used to find report files.",
    )
    parser.add_argument(
        "--top-k-operations",
        type=int,
        default=25,
        help="Number of top operations to show in plots and summary tables.",
    )
    parser.add_argument(
        "--max-annotation-len",
        type=int,
        default=18,
        help="Maximum text length used in annotated heatmaps.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def basename_from_run_dir(path_str: str) -> str:
    return Path(path_str).name


def parse_step_name(step_name: str) -> Tuple[Optional[int], str]:
    """Return (step_index, normalized_name_without_numeric_prefix)."""
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


def shorten(text: Optional[str], max_len: int) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def canonical_operation_name(normalized_name: str) -> str:
    """Remove only the leading numeric prefix. Keep tensor/patch suffixes.

    This is intentionally conservative because the paper often benefits from
    distinguishing operations such as cumsum_a, cumsum_b, patch_cumsum_a_9,
    patch_cumsum_b_25, etc.
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
        if normalized_name.startswith(("patch_sum", "patch_mean", "patch_std", "patch_var", "patch_norm", "patch_amax", "patch_amin")):
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
        "sqrt_", "rsqrt_", "log_", "log1p_", "exp_", "expm1_", "sin_", "cos_", "tan_",
        "arcsin_", "arccos_", "arctan_", "atan2_", "sinh_", "cosh_", "tanh_", "erf_",
        "sigmoid_", "relu_", "leaky_relu_", "softplus_", "hypot_",
    )):
        return "transcendental_nonlinear"

    if normalized_name.startswith((
        "abs_", "neg_", "sign_", "round_", "floor_", "ceil_", "trunc_", "frac_", "clamp_"
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
    """Higher-level grouping suitable for paper tables/figures."""
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


def load_reports(paths: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    pair_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for path in sorted(paths):
        try:
            report = load_json(path)
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"Could not read {path}: {exc}")
            continue

        if report.get("status") == "failed":
            reference_name = basename_from_run_dir(report.get("reference_dir", "unknown_reference"))
            candidate_name = basename_from_run_dir(report.get("candidate_dir", "unknown_candidate"))
            pair_rows.append(
                {
                    "report_path": str(path),
                    "reference": reference_name,
                    "candidate": candidate_name,
                    "status": "failed",
                    "error_message": report.get("error_message"),
                }
            )
            continue

        reference_name = basename_from_run_dir(report["reference_dir"])
        candidate_name = basename_from_run_dir(report["candidate_dir"])
        summary = report.get("summary", {})
        first_divergent_step = report.get("first_divergent_step")
        _, first_divergent_normalized = parse_step_name(first_divergent_step) if first_divergent_step else (None, None)

        pair_rows.append(
            {
                "report_path": str(path),
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
                "num_divergent_steps_in_report": len(report.get("divergent_steps", [])),
            }
        )

        for item in report.get("divergent_steps", []):
            step_name = item["step_name"]
            step_index, normalized_name = parse_step_name(step_name)
            step_rows.append(
                {
                    "report_path": str(path),
                    "reference": reference_name,
                    "candidate": candidate_name,
                    "step_name": step_name,
                    "step_index": step_index,
                    "operation": canonical_operation_name(normalized_name),
                    "family": operation_family(normalized_name),
                    "group": operation_group_for_paper(normalized_name),
                    "patch_size": extract_patch_size(normalized_name),
                    "precision_variant": precision_variant(normalized_name),
                    "exact_equal": item.get("exact_equal"),
                    "allclose": item.get("allclose"),
                    "same_shape": item.get("same_shape"),
                    "num_elements": item.get("num_elements"),
                    "num_unequal": item.get("num_unequal"),
                    "fraction_unequal": item.get("fraction_unequal"),
                    "max_abs_diff": item.get("max_abs_diff"),
                    "mean_abs_diff": item.get("mean_abs_diff"),
                    "rmse_abs_diff": item.get("rmse_abs_diff"),
                    "max_rel_diff": item.get("max_rel_diff"),
                    "mean_rel_diff": item.get("mean_rel_diff"),
                    "candidate_nan_count": item.get("candidate_nan_count"),
                    "candidate_inf_count": item.get("candidate_inf_count"),
                    "reference_nan_count": item.get("reference_nan_count"),
                    "reference_inf_count": item.get("reference_inf_count"),
                    "first_diff_index": json.dumps(item.get("first_diff_index")),
                    "first_diff_reference_value": item.get("first_diff_reference_value"),
                    "first_diff_candidate_value": item.get("first_diff_candidate_value"),
                }
            )

    return pd.DataFrame(pair_rows), pd.DataFrame(step_rows), warnings


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def aggregate_platform_role(pair_df: pd.DataFrame, role_column: str) -> pd.DataFrame:
    metric_cols = [
        "divergent_tensor_step_count",
        "exact_equal_tensor_step_count",
        "total_elements_compared",
        "total_unequal_elements",
        "unequal_element_rate",
    ]
    existing_metric_cols = [c for c in metric_cols if c in pair_df.columns]
    grouped = pair_df[pair_df["status"] == "ok"].groupby(role_column)[existing_metric_cols].agg(["mean", "median", "max", "min"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index().rename(columns={role_column: "platform"})
    return grouped.sort_values("platform")


def aggregate_by_operation(step_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        step_df.groupby(["operation", "family", "group", "precision_variant", "patch_size"], dropna=False)
        .agg(
            divergent_occurrences=("step_name", "count"),
            unique_platform_pairs=("report_path", "nunique"),
            mean_fraction_unequal=("fraction_unequal", "mean"),
            median_fraction_unequal=("fraction_unequal", "median"),
            max_fraction_unequal=("fraction_unequal", "max"),
            mean_max_abs_diff=("max_abs_diff", "mean"),
            max_max_abs_diff=("max_abs_diff", "max"),
            mean_max_rel_diff=("max_rel_diff", "mean"),
            max_max_rel_diff=("max_rel_diff", "max"),
            mean_num_unequal=("num_unequal", "mean"),
            max_num_unequal=("num_unequal", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(["divergent_occurrences", "mean_fraction_unequal", "max_max_abs_diff"], ascending=[False, False, False])


def aggregate_by_family(step_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        step_df.groupby(["family", "group", "precision_variant"], dropna=False)
        .agg(
            divergent_occurrences=("step_name", "count"),
            unique_platform_pairs=("report_path", "nunique"),
            unique_operations=("operation", "nunique"),
            mean_fraction_unequal=("fraction_unequal", "mean"),
            median_fraction_unequal=("fraction_unequal", "median"),
            max_fraction_unequal=("fraction_unequal", "max"),
            mean_max_abs_diff=("max_abs_diff", "mean"),
            max_max_abs_diff=("max_abs_diff", "max"),
            mean_max_rel_diff=("max_rel_diff", "mean"),
            max_max_rel_diff=("max_rel_diff", "max"),
            mean_num_unequal=("num_unequal", "mean"),
            max_num_unequal=("num_unequal", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(["divergent_occurrences", "mean_fraction_unequal"], ascending=[False, False])


def aggregate_patch_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    patch_df = step_df[step_df["patch_size"].notna()].copy()
    if patch_df.empty:
        return pd.DataFrame()
    grouped = (
        patch_df.groupby(["patch_size", "family", "operation", "precision_variant"], dropna=False)
        .agg(
            divergent_occurrences=("step_name", "count"),
            mean_fraction_unequal=("fraction_unequal", "mean"),
            max_fraction_unequal=("fraction_unequal", "max"),
            mean_max_abs_diff=("max_abs_diff", "mean"),
            max_max_abs_diff=("max_abs_diff", "max"),
            mean_max_rel_diff=("max_rel_diff", "mean"),
            max_max_rel_diff=("max_rel_diff", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(["patch_size", "divergent_occurrences"], ascending=[True, False])


def summarize_first_divergent(pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    by_operation = ok_df.groupby("first_divergent_operation", dropna=False).size().reset_index(name="count")
    by_operation = by_operation.sort_values("count", ascending=False)
    by_family = ok_df.groupby("first_divergent_family", dropna=False).size().reset_index(name="count")
    by_family = by_family.sort_values("count", ascending=False)
    return by_operation, by_family


def build_matrix(pair_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    matrix = ok_df.pivot(index="reference", columns="candidate", values=value_col)
    platforms = sorted(set(ok_df["reference"]).union(set(ok_df["candidate"])))
    matrix = matrix.reindex(index=platforms, columns=platforms)
    for platform in platforms:
        if platform in matrix.index and platform in matrix.columns:
            matrix.loc[platform, platform] = np.nan
    return matrix


def build_text_matrix(pair_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    matrix = ok_df.pivot(index="reference", columns="candidate", values=value_col)
    platforms = sorted(set(ok_df["reference"]).union(set(ok_df["candidate"])))
    matrix = matrix.reindex(index=platforms, columns=platforms)
    for platform in platforms:
        if platform in matrix.index and platform in matrix.columns:
            matrix.loc[platform, platform] = "-"
    return matrix


def symmetry_analysis(pair_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    pair_map = {
        (row.reference, row.candidate): row
        for row in ok_df.itertuples(index=False)
    }

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
    return pd.DataFrame(rows).sort_values(["abs_diff_unequal_rate", "abs_diff_divergent_steps"], ascending=[False, False])


def pair_presence_matrix(pair_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = pair_df[pair_df["status"] == "ok"].copy()
    ok_df["present"] = 1
    matrix = ok_df.pivot(index="reference", columns="candidate", values="present")
    platforms = sorted(set(ok_df["reference"]).union(set(ok_df["candidate"])))
    matrix = matrix.reindex(index=platforms, columns=platforms).fillna(0).astype(int)
    return matrix


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
            if math.isnan(val):
                text = "-"
            else:
                text = format(val, value_format)
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
    if matrix_values.empty:
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
            val = matrix_values.iloc[i, j]
            ax.text(j, i, shorten(str(val), max_annotation_len), ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Step index", rotation=-90, va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, top_k: int = 20, rotation: int = 45) -> None:
    if df.empty:
        return
    plot_df = df.head(top_k).copy()

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(plot_df)), 5.5))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, label_col: str, title: str, output_path: Path, top_k: int = 25) -> None:
    if df.empty:
        return
    plot_df = df.head(top_k).copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df[x_col].astype(float), plot_df[y_col].astype(float))
    for _, row in plot_df.iterrows():
        ax.annotate(str(row[label_col]), (row[x_col], row[y_col]), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_markdown_summary(
    pair_df: pd.DataFrame,
    step_df: pd.DataFrame,
    operation_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    first_divergent_by_family: pd.DataFrame,
    symmetry_df: pd.DataFrame,
    output_path: Path,
    top_k_operations: int,
) -> None:
    ok_pairs = pair_df[pair_df["status"] == "ok"].copy() if (not pair_df.empty and "status" in pair_df.columns) else pd.DataFrame()
    platforms = sorted(set(ok_pairs["reference"]).union(set(ok_pairs["candidate"])))

    lines: List[str] = []
    lines.append("# Cross-platform error analysis summary\n")
    lines.append(f"- Number of platforms observed: **{len(platforms)}**")
    lines.append(f"- Number of valid pairwise reports: **{len(ok_pairs)}**")
    lines.append(f"- Number of divergent step records: **{len(step_df)}**\n")

    if not ok_pairs.empty:
        lines.append("## Pairwise overview\n")
        lines.append(
            f"- Mean unequal element rate across pairs: **{ok_pairs['unequal_element_rate'].mean():.6f}**"
        )
        lines.append(
            f"- Median unequal element rate across pairs: **{ok_pairs['unequal_element_rate'].median():.6f}**"
        )
        lines.append(
            f"- Mean divergent tensor steps across pairs: **{ok_pairs['divergent_tensor_step_count'].mean():.2f}**"
        )
        most_severe_pair = ok_pairs.sort_values("unequal_element_rate", ascending=False).iloc[0]
        lines.append(
            f"- Most severe pair by unequal element rate: **{most_severe_pair['reference']} -> {most_severe_pair['candidate']}** "
            f"(**{most_severe_pair['unequal_element_rate']:.6f}**)\n"
        )

    if not first_divergent_by_family.empty:
        lines.append("## Most frequent first-divergent families\n")
        for _, row in first_divergent_by_family.head(10).iterrows():
            lines.append(f"- {row['first_divergent_family']}: {int(row['count'])} pairs")
        lines.append("")

    if not family_summary.empty:
        lines.append("## Most divergence-prone families\n")
        for _, row in family_summary.head(10).iterrows():
            lines.append(
                f"- {row['family']} ({row['precision_variant']}): "
                f"occurrences={int(row['divergent_occurrences'])}, "
                f"mean_fraction_unequal={row['mean_fraction_unequal']:.6f}, "
                f"max_abs_diff={row['max_max_abs_diff']:.6g}"
            )
        lines.append("")

    if not operation_summary.empty:
        lines.append("## Top operations by divergence occurrence\n")
        for _, row in operation_summary.head(top_k_operations).iterrows():
            lines.append(
                f"- {row['operation']}: occurrences={int(row['divergent_occurrences'])}, "
                f"mean_fraction_unequal={row['mean_fraction_unequal']:.6f}, "
                f"max_abs_diff={row['max_max_abs_diff']:.6g}, "
                f"family={row['family']}"
            )
        lines.append("")

    if not symmetry_df.empty:
        lines.append("## Most asymmetric platform pairs\n")
        for _, row in symmetry_df.head(10).iterrows():
            lines.append(
                f"- {row['platform_a']} <-> {row['platform_b']}: "
                f"|unequal_rate diff|={row['abs_diff_unequal_rate']:.6f}, "
                f"same first divergent family={bool(row['same_first_divergent_family'])}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    report_paths = list(args.input_dir.glob(args.glob))
    if not report_paths:
        raise FileNotFoundError(
            f"No report files found under {args.input_dir} using glob {args.glob!r}."
        )

    pair_df, step_df, warnings = load_reports(report_paths)

    if not pair_df.empty:
        save_csv(pair_df, args.output_dir / "pairwise_summary.csv")
    if not step_df.empty:
        save_csv(step_df, args.output_dir / "divergent_steps_long.csv")

    warnings_path = args.output_dir / "warnings.txt"
    warnings_path.write_text("\n".join(warnings) if warnings else "No warnings.", encoding="utf-8")

    ok_pairs = pair_df[pair_df["status"] == "ok"].copy() if (not pair_df.empty and "status" in pair_df.columns) else pd.DataFrame()

    # Platform role summaries.
    if not ok_pairs.empty:
        reference_role = aggregate_platform_role(ok_pairs, "reference")
        candidate_role = aggregate_platform_role(ok_pairs, "candidate")
        save_csv(reference_role, args.output_dir / "platform_summary_as_reference.csv")
        save_csv(candidate_role, args.output_dir / "platform_summary_as_candidate.csv")

    # Step-level summaries.
    if not step_df.empty:
        operation_summary = aggregate_by_operation(step_df)
        family_summary = aggregate_by_family(step_df)
        patch_summary = aggregate_patch_summary(step_df)
        save_csv(operation_summary, args.output_dir / "operation_summary.csv")
        save_csv(family_summary, args.output_dir / "family_summary.csv")
        if not patch_summary.empty:
            save_csv(patch_summary, args.output_dir / "patch_summary.csv")
    else:
        operation_summary = pd.DataFrame()
        family_summary = pd.DataFrame()
        patch_summary = pd.DataFrame()

    # First-divergent summaries.
    if not ok_pairs.empty:
        first_divergent_by_operation, first_divergent_by_family = summarize_first_divergent(ok_pairs)
        save_csv(first_divergent_by_operation, args.output_dir / "first_divergent_by_operation.csv")
        save_csv(first_divergent_by_family, args.output_dir / "first_divergent_by_family.csv")
    else:
        first_divergent_by_operation = pd.DataFrame()
        first_divergent_by_family = pd.DataFrame()

    # NxN matrices.
    if not ok_pairs.empty:
        pair_presence = pair_presence_matrix(ok_pairs)
        save_csv(pair_presence.reset_index().rename(columns={"reference": "platform"}), args.output_dir / "pair_presence_matrix.csv")

        unequal_rate_matrix = build_matrix(ok_pairs, "unequal_element_rate")
        divergent_steps_matrix = build_matrix(ok_pairs, "divergent_tensor_step_count")
        total_unequal_matrix = build_matrix(ok_pairs, "total_unequal_elements")
        first_divergent_step_matrix = build_text_matrix(ok_pairs, "first_divergent_operation")
        first_divergent_family_matrix = build_text_matrix(ok_pairs, "first_divergent_family")
        first_divergent_index_matrix = build_matrix(
            ok_pairs.assign(
                first_divergent_step_index=ok_pairs["first_divergent_step"].apply(
                    lambda x: parse_step_name(x)[0] if isinstance(x, str) else np.nan
                )
            ),
            "first_divergent_step_index",
        )

        unequal_rate_matrix.to_csv(args.output_dir / "matrix_unequal_element_rate.csv")
        divergent_steps_matrix.to_csv(args.output_dir / "matrix_divergent_tensor_step_count.csv")
        total_unequal_matrix.to_csv(args.output_dir / "matrix_total_unequal_elements.csv")
        first_divergent_step_matrix.to_csv(args.output_dir / "matrix_first_divergent_operation.csv")
        first_divergent_family_matrix.to_csv(args.output_dir / "matrix_first_divergent_family.csv")
        first_divergent_index_matrix.to_csv(args.output_dir / "matrix_first_divergent_step_index.csv")

        plot_numeric_heatmap(
            unequal_rate_matrix,
            "NxN unequal element rate",
            args.output_dir / "heatmap_unequal_element_rate.png",
            cmap="viridis",
            value_format=".3f",
        )
        plot_numeric_heatmap(
            divergent_steps_matrix,
            "NxN divergent tensor step count",
            args.output_dir / "heatmap_divergent_step_count.png",
            cmap="plasma",
            value_format=".0f",
        )
        plot_numeric_heatmap(
            first_divergent_index_matrix,
            "NxN first divergent step index",
            args.output_dir / "heatmap_first_divergent_step_index.png",
            cmap="magma",
            value_format=".0f",
        )
        plot_text_heatmap(
            first_divergent_family_matrix,
            first_divergent_index_matrix,
            "NxN first divergent family (color = step index)",
            args.output_dir / "heatmap_first_divergent_family.png",
            max_annotation_len=args.max_annotation_len,
            cmap="magma",
        )

    # Symmetry analysis.
    if not ok_pairs.empty:
        symmetry_df = symmetry_analysis(ok_pairs)
        if not symmetry_df.empty:
            save_csv(symmetry_df, args.output_dir / "symmetry_analysis.csv")
            plot_bar(
                symmetry_df,
                x_col="platform_a",
                y_col="abs_diff_unequal_rate",
                title="Directional asymmetry (absolute unequal-rate difference)",
                output_path=args.output_dir / "bar_directional_asymmetry.png",
                top_k=min(len(symmetry_df), 20),
            )
    else:
        symmetry_df = pd.DataFrame()

    # Family and operation plots.
    if not family_summary.empty:
        plot_bar(
            family_summary.sort_values("divergent_occurrences", ascending=False),
            x_col="family",
            y_col="divergent_occurrences",
            title="Divergent occurrences by family",
            output_path=args.output_dir / "bar_family_occurrences.png",
            top_k=min(len(family_summary), 20),
            rotation=35,
        )
        plot_bar(
            family_summary.sort_values("mean_fraction_unequal", ascending=False),
            x_col="family",
            y_col="mean_fraction_unequal",
            title="Mean fraction of unequal elements by family",
            output_path=args.output_dir / "bar_family_mean_fraction_unequal.png",
            top_k=min(len(family_summary), 20),
            rotation=35,
        )
        plot_scatter(
            family_summary.sort_values("divergent_occurrences", ascending=False),
            x_col="mean_fraction_unequal",
            y_col="max_max_abs_diff",
            label_col="family",
            title="Family-level divergence profile",
            output_path=args.output_dir / "scatter_family_profile.png",
            top_k=min(len(family_summary), 20),
        )

    if not operation_summary.empty:
        plot_bar(
            operation_summary.sort_values("divergent_occurrences", ascending=False),
            x_col="operation",
            y_col="divergent_occurrences",
            title="Top divergent operations by occurrence",
            output_path=args.output_dir / "bar_top_operations_by_occurrence.png",
            top_k=args.top_k_operations,
            rotation=75,
        )
        plot_bar(
            operation_summary.sort_values("mean_fraction_unequal", ascending=False),
            x_col="operation",
            y_col="mean_fraction_unequal",
            title="Top operations by mean fraction of unequal elements",
            output_path=args.output_dir / "bar_top_operations_by_mean_fraction_unequal.png",
            top_k=args.top_k_operations,
            rotation=75,
        )
        plot_bar(
            operation_summary.sort_values("max_max_abs_diff", ascending=False),
            x_col="operation",
            y_col="max_max_abs_diff",
            title="Top operations by maximum absolute difference",
            output_path=args.output_dir / "bar_top_operations_by_max_abs_diff.png",
            top_k=args.top_k_operations,
            rotation=75,
        )
        plot_scatter(
            operation_summary.sort_values("divergent_occurrences", ascending=False),
            x_col="mean_fraction_unequal",
            y_col="max_max_abs_diff",
            label_col="operation",
            title="Operation-level divergence profile",
            output_path=args.output_dir / "scatter_operation_profile.png",
            top_k=min(args.top_k_operations, len(operation_summary)),
        )

    if not patch_summary.empty:
        patch_family_summary = (
            patch_summary.groupby(["patch_size", "family", "precision_variant"], dropna=False)
            .agg(
                divergent_occurrences=("divergent_occurrences", "sum"),
                mean_fraction_unequal=("mean_fraction_unequal", "mean"),
                max_max_abs_diff=("max_max_abs_diff", "max"),
            )
            .reset_index()
            .sort_values(["patch_size", "divergent_occurrences"], ascending=[True, False])
        )
        save_csv(patch_family_summary, args.output_dir / "patch_family_summary.csv")

        for patch_size, group_df in patch_family_summary.groupby("patch_size"):
            safe_patch = str(int(patch_size)) if pd.notna(patch_size) else "unknown"
            plot_bar(
                group_df.sort_values("divergent_occurrences", ascending=False),
                x_col="family",
                y_col="divergent_occurrences",
                title=f"Patch-size {safe_patch}: divergent occurrences by family",
                output_path=args.output_dir / f"bar_patch_{safe_patch}_family_occurrences.png",
                top_k=min(len(group_df), 20),
                rotation=35,
            )

    create_markdown_summary(
        pair_df=pair_df,
        step_df=step_df,
        operation_summary=operation_summary,
        family_summary=family_summary,
        first_divergent_by_family=first_divergent_by_family,
        symmetry_df=symmetry_df,
        output_path=args.output_dir / "analysis_summary.md",
        top_k_operations=args.top_k_operations,
    )

    print(f"Found {len(report_paths)} report files.")
    print(f"Saved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
