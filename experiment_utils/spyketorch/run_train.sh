#!/bin/bash

# First set of ratios (without --augment)
for ratio in 0.01 0.02 0.1 0.25 0.5 0.8
do
    echo "Running training with ratio $ratio (without augment)..."
    python train.py --model mozafari2018 --epochs 4 8 100 --tensorboard --batch_size 1000 --ratio $ratio
done

# Second set of ratios (with --augment)
for ratio in 0.2 0.5 1
do
    echo "Running training with ratio $ratio (with augment)..."
    python train.py --model mozafari2018 --epochs 4 8 50 --tensorboard --batch_size 1000 --augment --ratio $ratio
done

echo "All trainings completed."
