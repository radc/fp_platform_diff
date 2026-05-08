EXPERIMENT_FILE=configs/experiment_phd.json

experiment_name=$(uname -n)
experiment_name="${experiment_name//[^[:alnum:]]/}"

echo "Running experiment: $experiment_name with config file: $EXPERIMENT_FILE"

run_name_reference="/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/PCEuler_gpu"


python main.py compare \
  --reference ${run_name_reference} \
  --candidate \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/PCEuler_cpu" \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/gn1001twccai_cpu" \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/gn1001twccai_gpu" \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/pcmoore_cpu" \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/pcmoore_gpu" \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/RuhansMacBookAirlocal_cpu" \
  "/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions/RuhansMacBookAirlocal_mps" \
  --format pt \
  --rtol 0.0 \
  --atol 0.0