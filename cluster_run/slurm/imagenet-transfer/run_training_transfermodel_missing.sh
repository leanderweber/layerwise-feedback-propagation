for config_file in ../../../configs/imagenet-transfer/cluster/*isic_vgg16_1.0*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer_transfermodel.sh $config_file
    sleep 0.5
done;

for config_file in ../../../configs/imagenet-transfer/cluster/*isic_vgg16_0.5*.yaml; do
    echo "Start training for $config_file"
    sbatch -p gpu3,gpu4 running_apptainer_transfermodel.sh $config_file
    sleep 0.5
done;
