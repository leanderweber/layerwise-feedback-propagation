for file in /data/cluster/users/lweber/reward-backprop/cifar100-cifar10-transfer/*yaml-output_data.tgz; do
    echo "Start tar reduction for $file"
    sbatch running_apptainer_tarreduction.sh $file
    sleep 0.5
done;
