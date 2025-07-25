export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_USE_CUDA_DSA=True
model_name=Qwen2.5-32B-Instruct
data_set=open_s11_32k_1k


output_path=/model_output/letz/sft/full/$data_set/$model_name
echo ./log/$data_set.log
mkdir -p $output_path
torchrun --nproc-per-node 8 --master_port 12345 train/sft.py \
    --block_size 32768 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 4 \
    --train_file_path ./data/$data_set \
    --model_name /gemini/data-1/$model_name \
    --warmup_ratio 0.05 \
    --deepspeed ../examples/deepspeed/ds_z3_offload_config.json \
    --bf16 \
    --logging_steps 5 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --output_dir $output_path \
    --save_steps 62 \
    --gradient_checkpointing > ./log/$data_set.log 2>&1 &

# wait
# done