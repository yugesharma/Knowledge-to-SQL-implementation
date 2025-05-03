#!/bin/bash
#SBATCH -N 1                      # allocate 1 compute node
#SBATCH -n 1                      # total number of tasks
#SBATCH --mem=64g                 # allocate 64 GB of memory (Llama2 13B is large)
#SBATCH -J "dpo_training"       # job name
#SBATCH -o dpo_training_%j.out        # output file
#SBATCH -e dpo_training_%j.err        # error file
#SBATCH -p short                  # partition (adjust if needed)
#SBATCH -t 24:00:00               # time limit (12 hours)
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH -C "H100|L40S"            # specify GPU types (A100 or H100 recommended)

# Load necessary modules
module load cuda/12.3.0/vuydybq
module load miniconda3
# Explicitly use the full path to conda for the specific environment


# Activate the specific environment


source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/etc/profile.d/conda.sh

# Activate the specific environment
conda activate /home/ysharma/.conda/envs/dellm

# Verify environment and torch installation

# Set up CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# wandb login
export WANDB_API_KEY="60c93797d275dde29587a3f96fc440212e7d7ff1"


# Run the training script
python model/bash.py \
    --model_name_or_path "meta-llama/Llama-2-13b-chat-hf" \
    --adapter_name_or_path "./output/llama2-13b-sft" \
    --dataset_dir "model" \
    --dataset "dpo_tokenized" \
    --template "llama2" \
    --split "train" \
    --output_dir "./output/llama2-13b-dpo" \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --fp16 \
    --gradient_checkpointing \
    --stage dpo \
    --finetuning_type lora \
    --lora_target "q_proj,v_proj" \
    --lora_rank 8 \
    --overwrite_output_dir
