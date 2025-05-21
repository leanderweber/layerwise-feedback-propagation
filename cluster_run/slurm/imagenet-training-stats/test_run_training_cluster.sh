for config_file in ../../../configs/imagenet-training-stats/cluster/cub_resnet18_0.005_lfp-epsilon_False_False_0.0001_onecyclelr_5110_transfermodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p testing running_apptainer_transfermodel.sh $config_file
    break
done;
