for config_file in ../../../configs/activationfuncs/cluster/cifar100_cifar-vgglike_sigmoid_1_lfp-epsilon_False_False_0.0_onecyclelr_7240_basemodel.yaml; do
    echo "Start training for $config_file"
    sbatch -p testing running_apptainer.sh $config_file
    break
done;
