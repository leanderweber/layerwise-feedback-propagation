model="$1"
data="$2"

mkdir -p /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/reward-backprop/resubmission-1-experiments/imagenet-training-stats-cluster-$model-$data

scp -r lweber@vca-gpu-headnode:/data/cluster/users/lweber/reward-backprop/imagenet-training-stats/${data}_${model}_*False_False_0.0001*smaller.tgz /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/reward-backprop/resubmission-1-experiments/imagenet-training-stats-cluster-$model-$data

for file in /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/reward-backprop/resubmission-1-experiments/imagenet-training-stats-cluster-$model-$data/*.tgz; do
    tar -zxf $file --one-top-level -C /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/reward-backprop/resubmission-1-experiments/imagenet-training-stats-cluster-$model-$data/;
    rm -rf $file;
done;
