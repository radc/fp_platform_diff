EXPERIMENT_FILE=configs/experiment_phd.json

experiment_name=$(uname -n)
experiment_name="${experiment_name//[^[:alnum:]]/}"

echo "Running experiment: $experiment_name with config file: $EXPERIMENT_FILE"

run_name="${experiment_name}_cpu"

echo "Running experiment with run name: $run_name in CPU mode..."
python main.py execute \
  --config ${EXPERIMENT_FILE} \
  --device cpu \
  --run-name $run_name \
  --operation-file operation.py


run_name="${experiment_name}_gpu"

echo "Running experiment with run name: $run_name in GPU mode..."
python main.py execute \
  --config ${EXPERIMENT_FILE} \
  --device cuda \
  --run-name $run_name \
  --operation-file operation.py

