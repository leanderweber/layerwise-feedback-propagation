for config_file in ../../../configs/nondifferentiable-criterion/cluster/*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer.sh $config_file
    sleep 1
done;
