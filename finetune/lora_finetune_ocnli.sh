formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

deepspeed --include localhost:0 --master_port 19888 finetune.py \
 --model_name_or_path openbmb/MiniCPM4-0.5B \
 --output_dir output/OCNLILoRA/$formatted_time/ \
 --train_data_path data/data_5k_without_system/train.json \
 --eval_data_path data/data_5k_without_system/dev.json \
 --model_max_length 1024 \
 --fp16 \
 --use_lora \
 --weight_decay 0.01 \
 --eval_strategy steps \
 --save_strategy steps \
 --seed 42 \
 --log_level info \
 --logging_strategy steps \
 --dataloader_pin_memory false \
 --remove_unused_columns false \
 --deepspeed_config configs/ds_config_zero2_offload.json \
 --learning_rate 1e-4 \
 --lr_scheduler_type constant \
 --max_steps 544 \
 --warmup_steps 30 \
 --per_device_train_batch_size 10 \
 --per_device_eval_batch_size 10 \
 --gradient_accumulation_steps 1 \
 --eval_steps 181 \
 --save_steps 272 \
 --logging_steps 27