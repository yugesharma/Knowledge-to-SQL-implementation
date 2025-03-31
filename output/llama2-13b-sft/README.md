---
license: other
base_model: meta-llama/Llama-2-13b-hf
tags:
- llama-factory
- lora
- generated_from_trainer
library_name: peft
model-index:
- name: llama2-13b-sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama2-13b-sft

This model is a fine-tuned version of [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) on the processed_train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.37.2
- Pytorch 2.1.2+cu121
- Datasets 3.4.1
- Tokenizers 0.15.2