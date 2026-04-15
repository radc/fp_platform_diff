# Floating-Point Cross-Platform Experiment Framework

This project helps investigate where floating-point differences appear across platforms,
devices, and software environments.

The workflow has three stages:

1. **Generate inputs**
   - Creates configurable tensors.
   - Saves them in `.pt`, `.bin`, and `.txt` formats.
   - Stores metadata for reproducibility.

2. **Execute operations**
   - Loads the generated tensors.
   - Runs a user-editable `operation.py` script.
   - Logs every intermediate tensor step.
   - Supports CPU and GPU execution through PyTorch.

3. **Compare runs**
   - Compares two execution folders step by step.
   - Reports the first divergent operation.
   - Computes absolute, relative, and exact-equality metrics.

## Project structure

```text
fp_platform_diff/
  main.py
  operation.py
  configs/
    experiment_example.json
  src/
    generate_inputs.py
    execute_ops.py
    compare_runs.py
    io_utils.py
    metadata.py
    ops_context.py
    comparators.py
```

## Quick start

### 1) Generate inputs

```bash
python main.py generate --config configs/experiment_example.json
```

### 2) Execute operations on CPU

```bash
python main.py execute \
  --config configs/experiment_example.json \
  --device cpu \
  --run-name linux_cpu \
  --operation-file operation.py
```

### 3) Execute operations on GPU

```bash
python main.py execute \
  --config configs/experiment_example.json \
  --device cuda:0 \
  --run-name windows_gpu \
  --operation-file operation.py
```

### 4) Compare two runs

```bash
python main.py compare \
  --reference runs/fp_cross_platform_demo/executions/linux_cpu \
  --candidate runs/fp_cross_platform_demo/executions/windows_gpu \
  --format pt \
  --rtol 0.0 \
  --atol 0.0
```

## Notes

- The `.bin` format is useful when you want a raw binary representation.
- The `.txt` format is useful when you want to inspect decimal serialization effects.
- The `.pt` format is convenient for PyTorch workflows.
- For strict reproducibility experiments, you may want to disable TF32 and enable deterministic algorithms when supported.

## Suggested experiment strategy

- Start with small tensors for debugging.
- Then scale to tens of millions of values.
- Run exactly the same input set across CPU and GPU.
- Compare step-by-step outputs to identify the first operation that diverges.
