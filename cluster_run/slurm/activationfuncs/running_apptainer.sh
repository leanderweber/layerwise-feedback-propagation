#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leander.weber@hhi.fraunhofer.de
#SBATCH --job-name=lfp-transfer
#SBATCH --output=lfp-transfer-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

TOKEN='TBD'
USER_URL="vigitlab.fe.hhi.de/lweber/"
REPO_NAME="reward-backprop"
BRANCH_NAME="resubmission-1"
DATA_SOURCE_DIR="/home/fe/lweber"

RESULT_STORAGE_DIR="/data/cluster/users/lweber/reward-backprop/activationfuncs/"
mkdir -p $RESULT_STORAGE_DIR

source "/etc/slurm/local_job_dir.sh"

bash ../download_repository.sh -u $USER_URL -r $REPO_NAME -b $BRANCH_NAME -t $TOKEN -p ${LOCAL_JOB_DIR}
echo "LOCAL DIR: ${LOCAL_JOB_DIR}"

fname_config=$(basename $@)

mkdir -p ${LOCAL_JOB_DIR}/data

echo "Copying Data..."
echo ${DATA_SOURCE_DIR}
cp -r ${DATA_SOURCE_DIR}/cifar10 ${LOCAL_JOB_DIR}/data
cp -r ${DATA_SOURCE_DIR}/cifar100 ${LOCAL_JOB_DIR}/data

echo $fname_config

echo "Start Training"
apptainer run --nv \
              --bind ${LOCAL_JOB_DIR}:/mnt \
              --bind /data/datapool/datasets/adience:/mnt/data/adience \
              --bind /data/datapool/datasets/ISIC2019:/mnt/data/isic \
              --bind /data/datapool3/datasets/cub_attributes/CUB_200_2011:/mnt/data/cub \
              --bind /data/datapool/datasets/food-11:/mnt/data/food11 \
              --bind /data/datapool/datasets/ImageNet-complete:/mnt/data/imagenet \
              ../../singularity/image_mini.sif bash /mnt/reward-backprop/cluster_run/slurm/activationfuncs/run.sh $fname_config
echo "Training finished"

echo "Copying results"
cd ${LOCAL_JOB_DIR}
tar -czf ${fname_config}-output_data.tgz output
cp -r ${fname_config}-output_data.tgz ${RESULT_STORAGE_DIR}
mv ${SLURM_SUBMIT_DIR}/lfp-transfer-${SLURM_JOB_ID}.out ${RESULT_STORAGE_DIR}/lfp-transfer-${fname_config}-${SLURM_JOB_ID}.out
echo "Results Copied"
echo "Done!"
