#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    /data/home/zfyang/home/oishi/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data/home/zfyang/home/oishi/model/Llama-2-7b-hf \
    --dataset theorem_alpaca \
    --dataset_dir /data/home/zfyang/home/oishi/LLaMA-Factory/data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir /data/home/zfyang/home/oishi/model/Llama-2-7b-hf-sft/Llama-2-7b-hf-ft-v1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --ddp_timeout 1800000 \
    --plot_loss \
    --bf16
