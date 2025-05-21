for file in /data/cluster/users/lweber/reward-backprop/imagenet-transfer/cub*yaml-output_data.tgz; do
    echo "Start tar reduction for $file"
    sbatch running_apptainer_tarreduction.sh $file
    sleep 0.5
done;

for file in /data/cluster/users/lweber/reward-backprop/imagenet-transfer/isic*yaml-output_data.tgz; do
    echo "Start tar reduction for $file"
    sbatch running_apptainer_tarreduction_isic.sh $file
    sleep 0.5
done;
