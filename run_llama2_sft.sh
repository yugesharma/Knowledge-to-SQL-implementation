#!/bin/bash
#SBATCH -N 1                      
#SBATCH -n 1                      
#SBATCH --mem=64g                 
#SBATCH -J "llama2-13b-sft"       
#SBATCH -o training_%j.out        
#SBATCH -e training_%j.err        
#SBATCH -p short                  
#SBATCH -t 12:00:00               
#SBATCH --gres=gpu:1              
#SBATCH -C "A100|H100"            

# Load necessary modules
module load cuda/12.3.0/vuydybq
module load miniconda3

# Activate the specific environment
source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/etc/profile.d/conda.sh

# Activate the specific environment
conda activate /home/yourUsername/.conda/envs/dellm #replace yourUsername with your Turing cluster username

# Set up CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# wandb login
export WANDB_API_KEY="Enter WAND API key for training metrics"


# Run the training script
python model/bash.py --model_name_or_path "meta-llama/Llama-2-13b-hf" \
    --dataset "processed_train" \
    --dataset_dir "model" \
    --template "llama2" \
    --split "train" \
    --cutoff_len 1024 \
    --output_dir "./output/llama2-13b-sft" \
    --do_train \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --fp16 \
    --gradient_checkpointing \
    --stage sft \
    --finetuning_type lora \
    --lora_target "q_proj,v_proj" \
    --overwrite_output_dir
