for config_file in ../../../configs/nondifferentiable-criterion/cluster/*isic*.yaml; do
    echo "Start training for $config_file"
    sbatch -p testing running_apptainer.sh $config_file
    sleep 10
done;
