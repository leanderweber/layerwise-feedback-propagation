for file in /data/cluster/users/lweber/reward-backprop/imagenet-transfer/cub*yaml-output_data.tgz; do
    echo "Start tar reduction intermediate for $file"
    sbatch running_apptainer_tarreduction_intermediate.sh $file
    sleep 0.5
done;

for file in /data/cluster/users/lweber/reward-backprop/imagenet-transfer/isic*yaml-output_data.tgz; do
    echo "Start tar reduction intermediate for $file"
    sbatch running_apptainer_tarreduction_intermediate_isic.sh $file
    sleep 0.5
done;
