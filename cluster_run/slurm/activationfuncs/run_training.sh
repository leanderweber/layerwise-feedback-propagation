for config_file in ../../../configs/activationfuncs/cluster/cifar10_*_elu_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;

for config_file in ../../../configs/activationfuncs/cluster/cifar10_*_relu_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;

for config_file in ../../../configs/activationfuncs/cluster/cifar100_*_silu_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;

for config_file in ../../../configs/activationfuncs/cluster/cifar100_*_step_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;

for config_file in ../../../configs/activationfuncs/cluster/cifar100_*_tanh_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;

for config_file in ../../../configs/activationfuncs/cluster/cifar100_*_sigmoid_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;

for config_file in ../../../configs/activationfuncs/cluster/cifar100_*_relu_*vanilla-gradient*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 300
done;
