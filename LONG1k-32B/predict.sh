# export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
model_name=Qwen2.5-32B-Instruct
for seed in 42
do
for test_data in gpqa_diamond
do
template=deepseek-qwen
# for train_data in open_s11_32k_1k
# do
train_data=open_s11_32k_2k
data_path=./data/${test_data}.json
# output_path=./output/$test_data/$model_name-$train_data
output_path=./output/$test_data/$model_name-$train_data-$seed
log_path=log/$test_data
mkdir -p $log_path
model_path=/model_output/letz/sft/full/$train_data/$model_name
# model_path=/model_output/letz/sft/full/open_s11_32k_0.5k/Qwen2.5-32B-Instruct/checkpoint-65
# model_path=/home/gpuall/model_output/letz/merged_models/$test_data/$model_name
# model_path=/gemini/data-1/$model_name
echo $log_path/$model_name-$train_data-$seed.log

export PYTHONPATH=$PYTHONPATH:$model_path
python predict.py \
    --model_path $model_path \
    --max_tokens 35768 \
    --template $template \
    --data_path $data_path \
    --output_path $output_path \
    --gpu_memory_utilization 0.95 \
    --gpu_num 2 \
    --seed $seed \
    --max_len 40960 > $log_path/$model_name-$train_data-$seed.log 2>&1 &
wait
done
wait
done
# wait
# done