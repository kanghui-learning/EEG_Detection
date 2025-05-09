#!/bin/bash

# Dataset configuration
dataset_name="Patient_15"
seed=42
gpu='1'
adj_type='train'
batch_size=512
num_epochs=20

# Define methods and learning rates to try
# methods=("CNN" "LSTM" "CNNLSTM" "DeepConvNet" "ResNet1D" "SimpleEEGNet" "STGCN" "AGCRN" "LVM" "BiGRU" "XGBoost")
methods=("STGCN")
# learning_rates=(0.00001 0.0001 0.001 0.01)
learning_rates=(0.0001)
smooth=false
denoise=true

# Create results directory if it doesn't exist
mkdir -p ./results

# Loop through each method
for method in "${methods[@]}"; do
    echo "Training with method: $method"
    
    # Loop through each learning rate
    for lr in "${learning_rates[@]}"; do
        echo "Training with learning rate: $lr"
        
        # Run training
        python baseline.py --dataset $dataset_name \
                          --method $method \
                          --seed $seed \
                          --gpu $gpu \
                          --adj_type $adj_type \
                          --batch_size $batch_size \
                          --learning_rate $lr \
                          --num_epochs $num_epochs \
                          $([ "$smooth" = true ] && echo "--smooth") \
                          $([ "$denoise" = true ] && echo "--denoise")
        
        # Add a small delay between runs to prevent GPU memory issues
        sleep 5
    done
    
    echo "Completed training for method: $method"
    echo "----------------------------------------"
done