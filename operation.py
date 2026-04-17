"""User-editable operation script.

This file is intentionally simple to edit.
The framework will call the `run(ctx)` function.

Use:
- `ctx.input(name)` to load an input tensor.
- `ctx.op(step_name, function, *args, **kwargs)` to execute and log one operation.

Every call to `ctx.op(...)` saves the intermediate result with a step number and step name.
"""

from __future__ import annotations

import torch


def run(ctx):
    """Run a sequence of floating-point operations.

    This version is designed to:
    - avoid NaN and Inf as much as possible,
    - keep step names unique,
    - include a richer set of operations for cross-platform analysis,
    - preserve readability so it is still easy to edit.
    """
    eps = 1e-6
    patch_sizes = (9, 25)

    a = ctx.input("a")
    b = ctx.input("b")
    c = ctx.input("c")
    d = ctx.input("d")
    e = ctx.input("e")
    f = ctx.input("f")
    g = ctx.input("g")
    h = ctx.input("h")

    return_dict = {}

    def record(step_name, function, *args, **kwargs):
        """Execute one step through the framework and store it in the return dict."""
        out = ctx.op(step_name, function, *args, **kwargs)
        return_dict[step_name] = out
        return out

    def safe_denominator(x: torch.Tensor) -> torch.Tensor:
        """Avoid zero and near-zero denominators while preserving sign."""
        return torch.where(
            x.abs() < eps,
            torch.where(x >= 0, torch.full_like(x, eps), torch.full_like(x, -eps)),
            x,
        )

    def safe_positive(x: torch.Tensor) -> torch.Tensor:
        """Force strictly positive values for log/sqrt/power-like operations."""
        return x.abs() + eps

    def safe_unit_interval(x: torch.Tensor) -> torch.Tensor:
        """Clamp into (-1, 1) for inverse trigonometric functions."""
        return torch.clamp(x, -1.0 + eps, 1.0 - eps)

    def safe_exp_input(x: torch.Tensor) -> torch.Tensor:
        """Clamp inputs to avoid overflow in exp/expm1."""
        return torch.clamp(x, -20.0, 20.0)

    def safe_tan_input(x: torch.Tensor) -> torch.Tensor:
        """Clamp inputs to a range that avoids extreme tan values."""
        return torch.clamp(x, -1.0, 1.0)

    # -------------------------------------------------------------------------
    # Basic element-wise arithmetic
    # -------------------------------------------------------------------------
    add_ab = record("add_ab", torch.add, a, b)
    add_cd = record("add_cd", torch.add, c, d)
    add_ef = record("add_ef", torch.add, e, f)
    add_gh = record("add_gh", torch.add, g, h)

    add_ab_cd = record("add_ab_cd", torch.add, add_ab, add_cd)
    add_ef_gh = record("add_ef_gh", torch.add, add_ef, add_gh)
    add_all_tree = record("add_all_tree", torch.add, add_ab_cd, add_ef_gh)

    add_abc = record("add_abc", torch.add, add_ab, c)
    add_abcd = record("add_abcd", torch.add, add_abc, d)
    add_abcde = record("add_abcde", torch.add, add_abcd, e)
    add_abcdef = record("add_abcdef", torch.add, add_abcde, f)
    add_abcdefg = record("add_abcdefg", torch.add, add_abcdef, g)
    add_abcdefgh = record("add_abcdefgh", torch.add, add_abcdefg, h)

    # Associativity tests
    left_assoc_abc = record("left_assoc_abc", lambda x, y, z: (x + y) + z, a, b, c)
    right_assoc_abc = record("right_assoc_abc", lambda x, y, z: x + (y + z), a, b, c)
    assoc_diff_abc = record("assoc_diff_abc", torch.sub, left_assoc_abc, right_assoc_abc)

    left_assoc_abcd = record(
        "left_assoc_abcd",
        lambda w, x, y, z: (((w + x) + y) + z),
        a,
        b,
        c,
        d,
    )
    tree_assoc_abcd = record(
        "tree_assoc_abcd",
        lambda w, x, y, z: (w + x) + (y + z),
        a,
        b,
        c,
        d,
    )
    assoc_diff_abcd = record(
        "assoc_diff_abcd",
        torch.sub,
        left_assoc_abcd,
        tree_assoc_abcd,
    )

    sub_ab = record("sub_ab", torch.sub, a, b)
    sub_cd = record("sub_cd", torch.sub, c, d)
    sub_ef = record("sub_ef", torch.sub, e, f)
    sub_gh = record("sub_gh", torch.sub, g, h)
    sub_ba = record("sub_ba", torch.sub, b, a)
    sub_dc = record("sub_dc", torch.sub, d, c)
    sub_fe = record("sub_fe", torch.sub, f, e)
    sub_hg = record("sub_hg", torch.sub, h, g)

    mul_ab = record("mul_ab", torch.mul, a, b)
    mul_cd = record("mul_cd", torch.mul, c, d)
    mul_ef = record("mul_ef", torch.mul, e, f)
    mul_gh = record("mul_gh", torch.mul, g, h)

    mul_ab_plus_mul_cd = record("mul_ab_plus_mul_cd", torch.add, mul_ab, mul_cd)
    mul_ef_plus_mul_gh = record("mul_ef_plus_mul_gh", torch.add, mul_ef, mul_gh)
    mul_all_tree = record("mul_all_tree", torch.add, mul_ab_plus_mul_cd, mul_ef_plus_mul_gh)

    mul_ac = record("mul_ac", torch.mul, a, c)
    mul_bd = record("mul_bd", torch.mul, b, d)
    mul_eg = record("mul_eg", torch.mul, e, g)
    mul_fh = record("mul_fh", torch.mul, f, h)

    div_ab = record("div_ab", torch.div, a, safe_denominator(b))
    div_cd = record("div_cd", torch.div, c, safe_denominator(d))
    div_ef = record("div_ef", torch.div, e, safe_denominator(f))
    div_gh = record("div_gh", torch.div, g, safe_denominator(h))

    div_ba = record("div_ba", torch.div, b, safe_denominator(a))
    div_dc = record("div_dc", torch.div, d, safe_denominator(c))
    div_fe = record("div_fe", torch.div, f, safe_denominator(e))
    div_hg = record("div_hg", torch.div, h, safe_denominator(g))

    reciprocal_a = record("reciprocal_a", torch.reciprocal, safe_denominator(a))
    reciprocal_b = record("reciprocal_b", torch.reciprocal, safe_denominator(b))

    # -------------------------------------------------------------------------
    # Power-like operations with safe domains
    # -------------------------------------------------------------------------
    square_a = record("square_a", lambda x: torch.pow(x, 2.0), a)
    square_b = record("square_b", lambda x: torch.pow(x, 2.0), b)

    cube_a = record("cube_a", lambda x: torch.pow(x, 3.0), a)
    cube_b = record("cube_b", lambda x: torch.pow(x, 3.0), b)

    pow_pos_a_1p5 = record("pow_pos_a_1p5", lambda x: torch.pow(safe_positive(x), 1.5), a)
    pow_pos_b_1p5 = record("pow_pos_b_1p5", lambda x: torch.pow(safe_positive(x), 1.5), b)

    # -------------------------------------------------------------------------
    # Rounding and sign-related operations
    # -------------------------------------------------------------------------
    abs_a = record("abs_a", torch.abs, a)
    abs_b = record("abs_b", torch.abs, b)

    neg_a = record("neg_a", torch.neg, a)
    neg_b = record("neg_b", torch.neg, b)

    sign_a = record("sign_a", torch.sign, a)
    sign_b = record("sign_b", torch.sign, b)

    round_a = record("round_a", torch.round, a)
    round_b = record("round_b", torch.round, b)

    floor_a = record("floor_a", torch.floor, a)
    floor_b = record("floor_b", torch.floor, b)

    ceil_a = record("ceil_a", torch.ceil, a)
    ceil_b = record("ceil_b", torch.ceil, b)

    trunc_a = record("trunc_a", torch.trunc, a)
    trunc_b = record("trunc_b", torch.trunc, b)

    frac_a = record("frac_a", torch.frac, a)
    frac_b = record("frac_b", torch.frac, b)

    clamp_a = record("clamp_a", lambda x: torch.clamp(x, -1.0, 1.0), a)
    clamp_b = record("clamp_b", lambda x: torch.clamp(x, -1.0, 1.0), b)

    # -------------------------------------------------------------------------
    # Root, logarithmic, exponential, and trigonometric operations
    # -------------------------------------------------------------------------
    sqrt_a = record("sqrt_a", torch.sqrt, safe_positive(a))
    sqrt_b = record("sqrt_b", torch.sqrt, safe_positive(b))

    rsqrt_a = record("rsqrt_a", torch.rsqrt, safe_positive(a))
    rsqrt_b = record("rsqrt_b", torch.rsqrt, safe_positive(b))

    log_a = record("log_a", torch.log, safe_positive(a))
    log_b = record("log_b", torch.log, safe_positive(b))

    log1p_a = record("log1p_a", torch.log1p, torch.abs(a))
    log1p_b = record("log1p_b", torch.log1p, torch.abs(b))

    exp_a = record("exp_a", torch.exp, safe_exp_input(a))
    exp_b = record("exp_b", torch.exp, safe_exp_input(b))

    expm1_a = record("expm1_a", torch.expm1, torch.clamp(a, -10.0, 10.0))
    expm1_b = record("expm1_b", torch.expm1, torch.clamp(b, -10.0, 10.0))

    sin_a = record("sin_a", torch.sin, a)
    sin_b = record("sin_b", torch.sin, b)

    cos_a = record("cos_a", torch.cos, a)
    cos_b = record("cos_b", torch.cos, b)

    tan_a = record("tan_a", torch.tan, safe_tan_input(a))
    tan_b = record("tan_b", torch.tan, safe_tan_input(b))

    arcsin_a = record("arcsin_a", torch.asin, safe_unit_interval(a))
    arcsin_b = record("arcsin_b", torch.asin, safe_unit_interval(b))

    arccos_a = record("arccos_a", torch.acos, safe_unit_interval(a))
    arccos_b = record("arccos_b", torch.acos, safe_unit_interval(b))

    arctan_a = record("arctan_a", torch.atan, a)
    arctan_b = record("arctan_b", torch.atan, b)

    atan2_ab = record("atan2_ab", torch.atan2, a, b)
    atan2_cd = record("atan2_cd", torch.atan2, c, d)

    sinh_a = record("sinh_a", torch.sinh, torch.clamp(a, -5.0, 5.0))
    sinh_b = record("sinh_b", torch.sinh, torch.clamp(b, -5.0, 5.0))

    cosh_a = record("cosh_a", torch.cosh, torch.clamp(a, -5.0, 5.0))
    cosh_b = record("cosh_b", torch.cosh, torch.clamp(b, -5.0, 5.0))

    tanh_a = record("tanh_a", torch.tanh, a)
    tanh_b = record("tanh_b", torch.tanh, b)

    erf_a = record("erf_a", torch.erf, a)
    erf_b = record("erf_b", torch.erf, b)

    sigmoid_a = record("sigmoid_a", torch.sigmoid, a)
    sigmoid_b = record("sigmoid_b", torch.sigmoid, b)

    relu_a = record("relu_a", torch.relu, a)
    relu_b = record("relu_b", torch.relu, b)

    leaky_relu_a = record("leaky_relu_a", lambda x: torch.where(x >= 0, x, 0.01 * x), a)
    leaky_relu_b = record("leaky_relu_b", lambda x: torch.where(x >= 0, x, 0.01 * x), b)

    softplus_a = record("softplus_a", lambda x: torch.log1p(torch.exp(torch.clamp(x, -20.0, 20.0))), a)
    softplus_b = record("softplus_b", lambda x: torch.log1p(torch.exp(torch.clamp(x, -20.0, 20.0))), b)

    hypot_ab = record("hypot_ab", torch.hypot, a, b)
    hypot_cd = record("hypot_cd", torch.hypot, c, d)

    # -------------------------------------------------------------------------
    # Fused / mixed operations
    # -------------------------------------------------------------------------
    addcmul_abc = record("addcmul_abc", torch.addcmul, a, b, c)
    addcdiv_abc = record("addcdiv_abc", torch.addcdiv, a, b, safe_denominator(c))

    lerp_ab_csig = record("lerp_ab_csig", torch.lerp, a, b, torch.sigmoid(c))
    lerp_cd_dsig = record("lerp_cd_dsig", torch.lerp, c, d, torch.sigmoid(e))

    # -------------------------------------------------------------------------
    # Global reductions
    # -------------------------------------------------------------------------
    cumsum_a = record("cumsum_a", torch.cumsum, a, 0)
    cumsum_b = record("cumsum_b", torch.cumsum, b, 0)

    cumsum_a_fp64 = record("cumsum_a_fp64", lambda x: torch.cumsum(x, dim=0, dtype=torch.float64), a)
    cumsum_b_fp64 = record("cumsum_b_fp64", lambda x: torch.cumsum(x, dim=0, dtype=torch.float64), b)

    sum_a = record("sum_a", torch.sum, a)
    sum_b = record("sum_b", torch.sum, b)

    sum_a_fp64 = record("sum_a_fp64", lambda x: torch.sum(x, dtype=torch.float64), a)
    sum_b_fp64 = record("sum_b_fp64", lambda x: torch.sum(x, dtype=torch.float64), b)

    mean_a = record("mean_a", torch.mean, a)
    mean_b = record("mean_b", torch.mean, b)

    std_a = record("std_a", lambda x: torch.std(x, unbiased=False), a)
    std_b = record("std_b", lambda x: torch.std(x, unbiased=False), b)

    var_a = record("var_a", lambda x: torch.var(x, unbiased=False), a)
    var_b = record("var_b", lambda x: torch.var(x, unbiased=False), b)

    norm_a = record("norm_a", lambda x: torch.linalg.vector_norm(x, ord=2), a)
    norm_b = record("norm_b", lambda x: torch.linalg.vector_norm(x, ord=2), b)

    max_a = record("max_a", torch.max, a)
    max_b = record("max_b", torch.max, b)

    min_a = record("min_a", torch.min, a)
    min_b = record("min_b", torch.min, b)

    # -------------------------------------------------------------------------
    # Repeated element-wise addition (not a true cumsum, but useful as a control)
    # -------------------------------------------------------------------------
    repeated_add_a_2x = record("repeated_add_a_2x", torch.add, a, a)
    current_repeated = repeated_add_a_2x
    for factor in range(3, 11):
        current_repeated = record(
            f"repeated_add_a_{factor}x",
            torch.add,
            current_repeated,
            a,
        )

    # -------------------------------------------------------------------------
    # Patch-based operations
    # -------------------------------------------------------------------------
    patch_sources = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "mul_ab": mul_ab,
        "mul_cd": mul_cd,
    }

    for patch_size in patch_sizes:
        for source_name, source_tensor in patch_sources.items():
            trimmed = record(
                f"{source_name}_trim_patch_{patch_size}",
                lambda x, ps=patch_size: x[: (x.numel() // ps) * ps],
                source_tensor,
            )

            patches = record(
                f"{source_name}_reshape_patch_{patch_size}",
                lambda x, ps=patch_size: x.reshape(-1, ps),
                trimmed,
            )

            patch_sum = record(
                f"patch_sum_{source_name}_{patch_size}",
                torch.sum,
                patches,
                1,
            )

            patch_sum_fp64 = record(
                f"patch_sum_fp64_{source_name}_{patch_size}",
                lambda x: torch.sum(x, dim=1, dtype=torch.float64),
                patches,
            )

            patch_mean = record(
                f"patch_mean_{source_name}_{patch_size}",
                torch.mean,
                patches,
                1,
            )

            patch_cumsum = record(
                f"patch_cumsum_{source_name}_{patch_size}",
                torch.cumsum,
                patches,
                1,
            )

            patch_cumsum_fp64 = record(
                f"patch_cumsum_fp64_{source_name}_{patch_size}",
                lambda x: torch.cumsum(x, dim=1, dtype=torch.float64),
                patches,
            )

            patch_std = record(
                f"patch_std_{source_name}_{patch_size}",
                lambda x: torch.std(x, dim=1, unbiased=False),
                patches,
            )

            patch_var = record(
                f"patch_var_{source_name}_{patch_size}",
                lambda x: torch.var(x, dim=1, unbiased=False),
                patches,
            )

            patch_norm_l1 = record(
                f"patch_norm_l1_{source_name}_{patch_size}",
                lambda x: torch.linalg.vector_norm(x, ord=1, dim=1),
                patches,
            )

            patch_norm_l2 = record(
                f"patch_norm_l2_{source_name}_{patch_size}",
                lambda x: torch.linalg.vector_norm(x, ord=2, dim=1),
                patches,
            )

            patch_amax = record(
                f"patch_amax_{source_name}_{patch_size}",
                lambda x: torch.amax(x, dim=1),
                patches,
            )

            patch_amin = record(
                f"patch_amin_{source_name}_{patch_size}",
                lambda x: torch.amin(x, dim=1),
                patches,
            )

            patch_softmax = record(
                f"patch_softmax_{source_name}_{patch_size}",
                lambda x: torch.softmax(x, dim=1),
                patches,
            )

    return return_dict