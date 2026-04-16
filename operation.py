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

    You can freely edit this function for your experiments.
    """
    a = ctx.input("a")
    b = ctx.input("b")
    c = ctx.input("c")
    d = ctx.input("d")
    e = ctx.input("e")
    f = ctx.input("f")
    g = ctx.input("g")
    h = ctx.input("h")

    # Basic arithmetic operations.
    add_ab = ctx.op("add_ab", torch.add, a, b)
    add_cd = ctx.op("add_ab", torch.add, c, d)
    add_ef = ctx.op("add_ab", torch.add, e, f)
    add_gh = ctx.op("add_ab", torch.add, g, h)
    add_ab_cd = ctx.op("add_ab_cd", torch.add, add_ab, add_cd)
    add_ef_gh = ctx.op("add_ef_gh", torch.add, add_ef, add_gh)
    add_all = ctx.op("add_all", torch.add, add_ab_cd, add_ef_gh)

    sub_ab = ctx.op("sub_ab", torch.sub, a, b)
    sub_cd = ctx.op("sub_cd", torch.sub, c, d)
    sub_ef = ctx.op("sub_ef", torch.sub, e, f)
    sub_gh = ctx.op("sub_gh", torch.sub, g, h)
    sub_ba = ctx.op("sub_ba", torch.sub, b, a)
    sub_dc = ctx.op("sub_dc", torch.sub, d, c)
    sub_fe = ctx.op("sub_fe", torch.sub, f, e)
    sub_hg = ctx.op("sub_hg", torch.sub, h, g)

    mul_ab = ctx.op("mul_ab", torch.mul, a, b)
    mul_cd = ctx.op("mul_cd", torch.mul, c, d)
    mul_ef = ctx.op("mul_ef", torch.mul, e, f)
    mul_gh = ctx.op("mul_gh", torch.mul, g, h)
    mul_ab_add_mul_cd = ctx.op("mul_ab_cd", torch.add, add_ab, add_cd)
    mul_ef_add_mul_gh = ctx.op("mul_ef_gh", torch.add, add_ef, add_gh)
    mul_add_all = ctx.op("mul_all", torch.add, mul_ab_add_mul_cd, mul_ef_add_mul_gh)

    mul_ac = ctx.op("mul_ac", torch.mul, a, c)
    mul_bd = ctx.op("mul_bd", torch.mul, b, d)
    mul_eg = ctx.op("mul_eg", torch.mul, e, g)
    mul_fh = ctx.op("mul_fh", torch.mul, f, h)

    div_ab = ctx.op("div_ab", torch.div, a, b)
    div_cd = ctx.op("div_cd", torch.div, c, d)
    div_ef = ctx.op("div_ef", torch.div, e, f)
    div_gh = ctx.op("div_gh", torch.div, g, h)

    div_ba = ctx.op("div_ba", torch.div, b, a)
    div_dc = ctx.op("div_dc", torch.div, d, c)
    div_fe = ctx.op("div_fe", torch.div, f, e)
    div_hg = ctx.op("div_hg", torch.div, h, g)


    cumsum_a = ctx.op("cumsum_a", torch.cumsum, a, 0)
    cumsum_b = ctx.op("cumsum_b", torch.cumsum, b, 0)
    cumsum_c = ctx.op("cumsum_c", torch.cumsum, c, 0)
    cumsum_d = ctx.op("cumsum_d", torch.cumsum, d, 0)
    cumsum_e = ctx.op("cumsum_e", torch.cumsum, e, 0)
    cumsum_f = ctx.op("cumsum_f", torch.cumsum, f, 0)
    cumsum_g = ctx.op("cumsum_g", torch.cumsum, g, 0)
    cumsum_h = ctx.op("cumsum_h", torch.cumsum, h, 0)


    # Round Operations
    round_a = ctx.op("round_a", torch.round, a)
    round_b = ctx.op("round_b", torch.round, b)
    round_c = ctx.op("round_c", torch.round, c)
    round_d = ctx.op("round_d", torch.round, d)
    round_e = ctx.op("round_e", torch.round, e)
    round_f = ctx.op("round_f", torch.round, f)
    round_g = ctx.op("round_g", torch.round, g)
    round_h = ctx.op("round_h", torch.round, h)

    floor_a = ctx.op("floor_a", torch.floor, a)
    floor_b = ctx.op("floor_b", torch.floor, b)
    floor_c = ctx.op("floor_c", torch.floor, c)
    floor_d = ctx.op("floor_d", torch.floor, d)
    floor_e = ctx.op("floor_e", torch.floor, e)
    floor_f = ctx.op("floor_f", torch.floor, f)
    floor_g = ctx.op("floor_g", torch.floor, g)
    floor_h = ctx.op("floor_h", torch.floor, h)

    ceil_a = ctx.op("ceil_a", torch.ceil, a)
    ceil_b = ctx.op("ceil_b", torch.ceil, b)
    ceil_c = ctx.op("ceil_c", torch.ceil, c)
    ceil_d = ctx.op("ceil_d", torch.ceil, d)
    ceil_e = ctx.op("ceil_e", torch.ceil, e)
    ceil_f = ctx.op("ceil_f", torch.ceil, f)
    ceil_g = ctx.op("ceil_g", torch.ceil, g)
    ceil_h = ctx.op("ceil_h", torch.ceil, h)
    
    sqrt_a = ctx.op("sqrt_a", torch.sqrt, torch.abs(a))
    sqrt_b = ctx.op("sqrt_b", torch.sqrt, torch.abs(b))
    sqrt_c = ctx.op("sqrt_c", torch.sqrt, torch.abs(c))
    sqrt_d = ctx.op("sqrt_d", torch.sqrt, torch.abs(d))
    sqrt_e = ctx.op("sqrt_e", torch.sqrt, torch.abs(e))
    sqrt_f = ctx.op("sqrt_f", torch.sqrt, torch.abs(f))
    sqrt_g = ctx.op("sqrt_g", torch.sqrt, torch.abs(g))
    sqrt_h = ctx.op("sqrt_h", torch.sqrt, torch.abs(h))    

    return_dict =  {
        "add_ab": add_ab,
        "add_cd": add_cd,
        "add_ef": add_ef,
        "add_gh": add_gh,
        "add_ab_cd": add_ab_cd,
        "add_ef_gh": add_ef_gh,
        "add_all": add_all,

        "sub_ab": sub_ab,
        "sub_cd": sub_cd,
        "sub_ef": sub_ef,
        "sub_gh": sub_gh,
        "sub_ba": sub_ba,
        "sub_dc": sub_dc,
        "sub_fe": sub_fe,
        "sub_hg": sub_hg,
        
        "mul_ab": mul_ab,
        "mul_cd": mul_cd,
        "mul_ef": mul_ef,
        "mul_gh": mul_gh,

        "mul_ab_add_mul_cd": mul_ab_add_mul_cd,
        "mul_ef_add_mul_gh": mul_ef_add_mul_gh,
        "mul_add_all": mul_add_all,

        "mul_ac": mul_ac,
        "mul_bd": mul_bd,
        "mul_eg": mul_eg,
        "mul_fh": mul_fh,

        "div_ab": div_ab,
        "div_cd": div_cd,
        "div_ef": div_ef,
        "div_gh": div_gh,
        "div_ba": div_ba,
        "div_dc": div_dc,
        "div_fe": div_fe,
        "div_hg": div_hg,
        
        "cumsum_a": cumsum_a,
        "cumsum_b": cumsum_b,
        "cumsum_c": cumsum_c,
        "cumsum_d": cumsum_d,
        "cumsum_e": cumsum_e,
        "cumsum_f": cumsum_f,
        "cumsum_g": cumsum_g,
        "cumsum_h": cumsum_h,        

        "round_a": round_a,
        "round_b": round_b,
        "round_c": round_c,
        "round_d": round_d,
        "round_e": round_e,
        "round_f": round_f,
        "round_g": round_g,
        "round_h": round_h,
        "floor_a": floor_a,
        "floor_b": floor_b,
        "floor_c": floor_c,
        "floor_d": floor_d,
        "floor_e": floor_e,
        "floor_f": floor_f,
        "floor_g": floor_g,
        "floor_h": floor_h,
        "ceil_a": ceil_a,
        "ceil_b": ceil_b,
        "ceil_c": ceil_c,
        "ceil_d": ceil_d,
        "ceil_e": ceil_e,
        "ceil_f": ceil_f,
        "ceil_g": ceil_g,
        "ceil_h": ceil_h,
                
        "sqrt_a": sqrt_a,
        "sqrt_b": sqrt_b,
        "sqrt_c": sqrt_c,
        "sqrt_d": sqrt_d,
        "sqrt_e": sqrt_e,
        "sqrt_f": sqrt_f,
        "sqrt_g": sqrt_g,
        "sqrt_h": sqrt_h,
    }
    
    for patch_size in [9, 25]:
        patch_size = 9
        a_trim = ctx.op(
            "a_trim_for_patch_ops",
            lambda x: x[: (x.numel() // patch_size) * patch_size],
            a,
        )
        a_patches = ctx.op(
            "a_reshaped_to_patches",
            lambda x: x.reshape(-1, patch_size),
            a_trim,
        )    
        patch_cumsum_a = ctx.op(f"patch_cumsum_a_{patch_size}", torch.cumsum, a_patches, 1)
        return_dict[f"patch_cumsum_a_{patch_size}"] = patch_cumsum_a

        b_trim = ctx.op(
            "b_trim_for_patch_ops",
            lambda x: x[: (x.numel() // patch_size) * patch_size],
            b,
        )
        b_patches = ctx.op(
            "b_reshaped_to_patches",
            lambda x: x.reshape(-1, patch_size),
            b_trim,
        )    
        patch_cumsum_b = ctx.op(f"patch_cumsum_b_{patch_size}", torch.cumsum, b_patches, 1)
        return_dict[f"patch_cumsum_b_{patch_size}"] = patch_cumsum_b

        c_trim = ctx.op(
            "c_trim_for_patch_ops",
            lambda x: x[: (x.numel() // patch_size) * patch_size],
            c,
        )
        c_patches = ctx.op(
            "c_reshaped_to_patches",
            lambda x: x.reshape(-1, patch_size),
            c_trim,
        )    
        patch_cumsum_c = ctx.op(f"patch_cumsum_c_{patch_size}", torch.cumsum, c_patches, 1)
        return_dict[f"patch_cumsum_c_{patch_size}"] = patch_cumsum_c

        d_trim = ctx.op(
            "d_trim_for_patch_ops",
            lambda x: x[: (x.numel() // patch_size) * patch_size],
            d,
        )
        d_patches = ctx.op(
            "d_reshaped_to_patches",
            lambda x: x.reshape(-1, patch_size),
            d_trim,
        )
        patch_cumsum_d = ctx.op(f"patch_cumsum_d_{patch_size}", torch.cumsum, d_patches, 1)
        return_dict[f"patch_cumsum_d_{patch_size}"] = patch_cumsum_d

        e_trim = ctx.op(
            "e_trim_for_patch_ops",
            lambda x: x[: (x.numel() // patch_size) * patch_size],
            e,
        )
        e_patches = ctx.op(
            "e_reshaped_to_patches",
            lambda x: x.reshape(-1, patch_size),
            e_trim,
        )
        patch_cumsum_e = ctx.op(f"patch_cumsum_e_{patch_size}", torch.cumsum, e_patches, 1)
        return_dict[f"patch_cumsum_e_{patch_size}"] = patch_cumsum_e

        f_trim = ctx.op(
            "f_trim_for_patch_ops",
            lambda x: x[: (x.numel() // patch_size) * patch_size],
            f,
        )

        f_patches = ctx.op(
            "f_reshaped_to_patches",
            lambda x: x.reshape(-1, patch_size),
            f_trim,
        )
        patch_cumsum_f = ctx.op(f"patch_cumsum_f_{patch_size}", torch.cumsum, f_patches, 1)
        return_dict[f"patch_cumsum_f_{patch_size}"] = patch_cumsum_f

        mul_ab_trim = ctx.op("mul_ac_trim", torch.mul, a_trim, b_trim)
        mul_ab_patches = ctx.op(
            "mul_ac_patches",
            lambda x: x.reshape(-1, patch_size),
            mul_ab_trim,
        )
        patch_cumsum_mul_ab = ctx.op(f"patch_cumsum_mul_ab_{patch_size}", torch.cumsum, mul_ab_patches, 1)
        return_dict[f"patch_cumsum_mul_ab_{patch_size}"] = patch_cumsum_mul_ab


        mul_cd_trim = ctx.op("mul_cd_trim", torch.mul, c_trim, d_trim)
        mul_cd_patches = ctx.op(
            "mul_cd_patches",
            lambda x: x.reshape(-1, patch_size),
            mul_cd_trim,
        )
        patch_cumsum_mul_cd = ctx.op(f"patch_cumsum_mul_cd_{patch_size}", torch.cumsum, mul_cd_patches, 1)   
        return_dict[f"patch_cumsum_mul_cd_{patch_size}"] = patch_cumsum_mul_cd

        mul_ef_trim = ctx.op("mul_ef_trim", torch.mul, e_trim, f_trim)
        mul_ef_patches = ctx.op(
            "mul_ef_patches",
            lambda x: x.reshape(-1, patch_size),
            mul_ef_trim,
        )
        patch_cumsum_mul_ef = ctx.op(f"patch_cumsum_mul_ef_{patch_size}", torch.cumsum, mul_ef_patches, 1)
        return_dict[f"patch_cumsum_mul_ef_{patch_size}"] = patch_cumsum_mul_ef

    return return_dict
