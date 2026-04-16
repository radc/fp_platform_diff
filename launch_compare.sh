EXPERIMENT_FILE=configs/experiment_phd.json

experiment_name=$(uname -n)
experiment_name="${experiment_name//[^[:alnum:]]/}"

echo "Running experiment: $experiment_name with config file: $EXPERIMENT_FILE"

run_name_reference="${experiment_name}_cpu"
run_name_candidate="${experiment_name}_gpu"

python main.py compare \
  --reference runs/fp_crossplatform/executions/${run_name_reference} \
  --candidate runs/fp_crossplatform/executions/${run_name_candidate} \
  --format pt \
  --rtol 0.0 \
  --atol 0.0