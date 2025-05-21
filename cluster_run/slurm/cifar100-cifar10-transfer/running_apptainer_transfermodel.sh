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

RESULT_STORAGE_DIR="/data/cluster/users/lweber/reward-backprop/cifar100-cifar10-transfer/"
mkdir -p $RESULT_STORAGE_DIR

source "/etc/slurm/local_job_dir.sh"

bash ../download_repository.sh -u $USER_URL -r $REPO_NAME -b $BRANCH_NAME -t $TOKEN -p ${LOCAL_JOB_DIR}
echo "LOCAL DIR: ${LOCAL_JOB_DIR}"

fname_config=$(basename $@)

mkdir -p ${LOCAL_JOB_DIR}/data

echo $fname_config

splitfname=(${fname_config//_/ })
seed=${splitfname[6]}

echo $seed

basetar=cifar100_0.05_vanilla-gradient_True_0.0_onecyclelr_${seed}_basemodel.yaml-output_data

cp ${RESULT_STORAGE_DIR}/${basetar}.tgz $LOCAL_JOB_DIR
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/${basetar}.tgz
rm -rf ${LOCAL_JOB_DIR}/output/${seed}/ckpts/*base-model-ep*.pt

dir ${LOCAL_JOB_DIR}/output/${seed}/ckpts/

echo "Copying Data..."
cp $DATA_SOURCE_DIR/cifar10 ${LOCAL_JOB_DIR}/data
cp $DATA_SOURCE_DIR/cifar100 ${LOCAL_JOB_DIR}/data

echo "Start Training"
apptainer run --nv \
              --bind ${LOCAL_JOB_DIR}:/mnt \
              ../../singularity/image_mini.sif bash /mnt/reward-backprop/cluster_run/slurm/cifar100-cifar10-transfer/run.sh $fname_config
echo "Training finished"

echo "Copying results"
cd ${LOCAL_JOB_DIR}
tar -czf ${fname_config}-output_data.tgz output
cp -r ${fname_config}-output_data.tgz ${RESULT_STORAGE_DIR}
mv ${SLURM_SUBMIT_DIR}/lfp-transfer-${SLURM_JOB_ID}.out ${RESULT_STORAGE_DIR}/lfp-transfer-${fname_config}-${SLURM_JOB_ID}.out
echo "Results Copied"
echo "Done!"
