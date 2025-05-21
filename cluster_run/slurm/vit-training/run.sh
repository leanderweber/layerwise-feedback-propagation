mkdir -p /mnt/output
mkdir -p /mnt/input
cd /mnt/reward-backprop

echo "STARTING JOB $@"

python3 -m run_resubmission1_experiment_v2 --config_file "configs/vit-training/cluster/$@"
