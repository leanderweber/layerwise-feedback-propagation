for config_file in ../../../configs/cifar100-cifar10-transfer/cluster/*0.005_lfp-epsilon_True_0.0_onecyclelr*_transfermodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer_transfermodel.sh $config_file
    sleep 0.5
done;

for config_file in ../../../configs/cifar100-cifar10-transfer/cluster/*1e-05_vanilla-gradient_True_0.0_onecyclelr*_transfermodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer_transfermodel.sh $config_file
    sleep 0.5
done;
