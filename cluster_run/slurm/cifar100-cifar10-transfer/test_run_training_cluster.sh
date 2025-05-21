for config_file in ../../../configs/cifar100-cifar10-transfer/cluster/cifar10_0.05_lfp-epsilon_False_0.0_onecyclelr_5628_basemodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p testing running_apptainer_transfermodel.sh $config_file
    break
done;
