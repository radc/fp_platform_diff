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

    # Basic arithmetic operations.
    add_ab = ctx.op("add_ab", torch.add, a, b)
    expr1 = ctx.op("div_add_ab_by_c", torch.div, add_ab, c)

    round_a = ctx.op("round_a", torch.round, a)
    mul_bc = ctx.op("mul_bc", torch.mul, b, c)
    expr2 = ctx.op("round_a_plus_mul_bc", torch.add, round_a, mul_bc)

    expr3 = ctx.op("div_a_by_b", torch.div, a, b)

    # Reduction and accumulation operations.
    sum_a = ctx.op("sum_a", torch.sum, a)
    cumsum_a = ctx.op("cumsum_a", torch.cumsum, a, 0)

    # A non-associativity example.
    left_assoc = ctx.op("left_assoc", torch.add, torch.add(a, b), c)
    right_assoc = ctx.op("right_assoc", torch.add, a, torch.add(b, c))
    assoc_diff = ctx.op("assoc_difference", torch.sub, left_assoc, right_assoc)

    return {
        "expr1": expr1,
        "expr2": expr2,
        "expr3": expr3,
        "sum_a": sum_a,
        "cumsum_a": cumsum_a,
        "assoc_diff": assoc_diff,
    }
