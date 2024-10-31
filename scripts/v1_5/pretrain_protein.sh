#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat


########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########
#model_name_or_path ./checkpoints/$MODEL_VERSION \
#--protein_sequence_folder ./playground/data/protein_data \


#Meta/Meta-Llama3.1-8B-Instruct
#lmsys/vicuna-7b-v1.5

# Meta's Llama 3.1 models & evals gated group (17 repositories)
# meta-llama/Meta-Llama-3.1-8B
# meta-llama/Meta-Llama-3.1-70B
# meta-llama/Meta-Llama-3.1-405B
# meta-llama/Meta-Llama-3.1-405B-Instruct-FP8
# meta-llama/Meta-Llama-3.1-8B-Instruct
# meta-llama/Meta-Llama-3.1-70B-Instruct
# meta-llama/Meta-Llama-3.1-405B-Instruct
# meta-llama/Meta-Llama-3.1-405B-FP8
# meta-llama/Prompt-Guard-86M
# meta-llama/Meta-Llama-3.1-405B-Instruct-evals


# meta-llama/Llama-3.2-1B
#meta-llama/Llama-3.2-1B-Instruct
#

# --gradient_accumulation_steps 2
#--torch_empty_cache_steps 1000
# --gradient_checkpointing False
# --dataloader_num_workers 2


#./scripts/v1_5/pretrain_protein.sh > output.txt 2>&1
deepspeed llava/train/train_mem_protein.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5\
    --version plain \
    --data_path ./playground/data/protein_data_with_amino.json \
    --protein_encoder facebook/esm2_t6_8M_UR50D \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True  \
    --mm_protein_select_layer -2 \
    --mm_protein_select_feature cls \
    --mm_use_protein_start_end False \
    --mm_use_protein_segment_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava1.5-vicuna-13b-pretrain-5epochs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 
