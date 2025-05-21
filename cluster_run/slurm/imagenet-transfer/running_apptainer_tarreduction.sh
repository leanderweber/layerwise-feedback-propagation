#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leander.weber@hhi.fraunhofer.de
#SBATCH --job-name=lfp-transfer
#SBATCH --output=lfp-transfer-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G

TOKEN='TBD'
USER_URL="vigitlab.fe.hhi.de/lweber/"
REPO_NAME="reward-backprop"
BRANCH_NAME="resubmission-1"

RESULT_STORAGE_DIR="/data/cluster/users/lweber/reward-backprop/imagenet-transfer/"
mkdir -p $RESULT_STORAGE_DIR

source "/etc/slurm/local_job_dir.sh"

echo "LOCAL DIR: ${LOCAL_JOB_DIR}"

fname=$(basename $@ .tgz)

splitfname=(${fname//_/ })
seed=${splitfname[7]}

echo $fname
echo $seed

cp ${RESULT_STORAGE_DIR}/${fname}.tgz $LOCAL_JOB_DIR
tar -C ${LOCAL_JOB_DIR} -zxvf ${LOCAL_JOB_DIR}/${fname}.tgz

rm -rf ${LOCAL_JOB_DIR}/${fname}.tgz
rm -rf ${LOCAL_JOB_DIR}/output/${seed}/wandb
rm -rf ${LOCAL_JOB_DIR}/output/${seed}/ckpts/transfer-model-ep*.pt

cd ${LOCAL_JOB_DIR}
tar -czf ${fname}-smaller.tgz output
cp -r ${fname}-smaller.tgz ${RESULT_STORAGE_DIR}
echo "Results Copied"
echo "Done!"
