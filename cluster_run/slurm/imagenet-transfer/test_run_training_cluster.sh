for config_file in ../../../configs/imagenet-transfer/cluster/adience_resnet18_0.1_lfp-epsilon_True_0.0_onecyclelr_5110_transfermodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p testing running_apptainer_transfermodel.sh $config_file
    break
done;
