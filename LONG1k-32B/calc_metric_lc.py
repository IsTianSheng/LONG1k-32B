from vllm import LLM,SamplingParams
import torch
import json
import re
import os



model_path = '/gemini/data-1/Qwen2.5-32B-Instruct'
llm = LLM(model=model_path,tensor_parallel_size=2, trust_remote_code=True,max_model_len=32768,gpu_memory_utilization=0.9,enable_lora=True)
sampling_params = SamplingParams(max_tokens=32678, temperature=0.9, top_p=0.7)

import os
import json
def get_data4json(output_path):
    pres = []
    refs = []
    problems = []
    for line in json.load(open(output_path,encoding='utf')):
        problem = line['input'].strip().strip('\n')
        pre = line['pre'].strip().strip('\n')
        ref = line['ref'].strip().strip('\n')
        problems.append(problem)
        pres.append(pre)
        refs.append(ref)
    return refs,pres,problems


for task in ['gpqa_diamond']:
    for train_data in ['Qwen2.5-32B-Instruct-all_constant_length_scaling_1500-100-0','Qwen2.5-32B-Instruct-all_constant_length_scaling_1500-150-0','Qwen2.5-32B-Instruct-all_constant_length_scaling_1500-200-0','Qwen2.5-32B-Instruct-all_constant_length_scaling_1500-250-0','Qwen2.5-32B-Instruct-all_constant_length_scaling_1500-300-0']:
        for model in ['Qwen2.5-32B-Instruct']:
            eval_result = []
            # model = f"{model}-{train_data}"
            output_path = f'./output/{task}/{train_data}/predict.json'
            refs,pres,questions = get_data4json(output_path)
            data = json.load(open(output_path))
            prompts = []
            scores = 0
            pattern = r'\[\[([0-9]+)\]\]'
            for i in range(len(refs)):
                ref = refs[i].split('\n')[-1]
                pre = "".join(pres[i].split('\n')[-5:])
                if task!='gpqa_diamond':
                    prompt='''现在我会向你输入一段数学问题，问题的标准答案，以及由某个大语言模型对该问题的输出，你需要从大语言模型的输出中提取出答案文本，然后再和标准答案进行对比，判断是否答对了这个问题。如果答对了判断为正确得1分，否则就是错误的得0分，问题是:{}\n标准答案是:{}\n模型输出是:{}\n,请你先从模型的输出文本中提取出最终答案，然后再对比标准答案看是否作答准确，你的回答模板是：问题是xxx\n标准的答案是xxx\n从模型输出文本中提取的答案是xxx\n，所以是正确/错误的，\n\n因此最终得[[0]]或[[1]]分。注意得分一定要使用两个中括号括起来。如果不能从模型输出中提取出答案，可以直接认为错误。注意有时模型输出的形式可能与标准答案形式有所差别，如发现经过简单换算或者添加单位后二者相等，则同样认为正确。'''.format(questions[i],ref,pre)
                else:
                    prompt='''现在我会向你输入一段涉及数学计算的科学问题，问题的标准答案，以及由某个大语言模型对该问题的输出，你需要从大语言模型的输出中提取出答案文本，然后再和标准答案进行对比，判断是否答对了这个问题。如果答对了判断为正确得1分，否则就是错误的得0分，问题是:{}\n标准答案是:{}\n模型输出是:{}\n,请你先从模型的输出文本中提取出最终答案，然后再对比标准答案看是否作答准确，你的回答模板是：问题是xxx\n标准的答案是xxx\n从模型输出文本中提取的答案是xxx\n，所以是正确/错误的，\n\n因此最终得[[0]]或[[1]]分。注意得分一定要使用两个中括号括起来。如果不能从模型输出中提取出答案，可以直接认为错误。注意有时模型输出的形式可能与标准答案形式有所差别，如发现经过简单换算或者添加单位后二者相等，则同样认为正确。'''.format(questions[i],ref,pre)
                prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(prompt)
                prompts.append(prompt)
            outputs = llm.generate(prompts, sampling_params)
            for i in range(len(outputs)):
                generated_text = outputs[i].outputs[0].text
                matches = re.findall(pattern, generated_text)
                if matches:
                    score = int(matches[0])
                else:
                    score = 0
                data[i]["score"] = score
                scores += score
            json.dump(data,open(output_path,'w'),ensure_ascii=False,indent=2)
            accuracy = scores / len(refs)
            print(output_path)
            print("Accuracy:", accuracy)
            print("score:", scores)
            print("-"*20)