mkdir -p /mnt/output
mkdir -p /mnt/input
cd /mnt/reward-backprop

echo "STARTING JOB $@"

python3 -m run_resubmission1_experiment --config_file "configs/imagenet-training-stats/cluster/$@"
