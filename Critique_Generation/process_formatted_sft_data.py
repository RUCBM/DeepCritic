import pandas as pd
import re
import json
import random

from transformers import AutoTokenizer
import numpy as np
random.seed(42)
model_path = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)

filename = "data/prm800k/phase2_train_final_critique.jsonl"
data_list = [json.loads(l) for l in open(filename, "r")]

filtered_data_list = []
correct_sample_no_correction, correct_sample_has_correction = [], []
for data in data_list:
    if -1 in data['critic_labels']:
        filtered_data_list.append(data)
    else:
        if False in data["critique_correctness"]:
            correct_sample_has_correction.append(data)
        else:
            correct_sample_no_correction.append(data)

random.shuffle(correct_sample_no_correction)
print(len(filtered_data_list))
print(len(correct_sample_no_correction))
print(len(correct_sample_has_correction))
# we suggest limiting the number of critiques for correct solutions for balanced performance
filtered_data_list = filtered_data_list + correct_sample_has_correction + correct_sample_no_correction[:250]
# Save filtered data to jsonl file
with open('data/prm800k/phase2_train_final_critque_filtered.jsonl', 'w') as f:
    for item in filtered_data_list:
        f.write(json.dumps(item) + '\n')

data_list = filtered_data_list

# print(data_list[0])
print(len(data_list))
file = open("ProcessBench/code/templates/critique_template_new.txt", "r")
prompt_template = file.read()
file.close()
new_data_list = []
response_len_list = []
label_len_dict = {}  
count = 0
correct_steps_no_correction, correct_steps_has_correction = 0, 0
wrong_steps_no_correction, wrong_steps_has_correction = 0, 0
label_distribution = {}
for data in data_list:
    new_data = {}
    steps = data['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        if "Step " not in step['text']:
            step = "Step {}: ".format((sdx + 1)) + step['text']
        tagged_response = tagged_response + step + "\n\n"
    tagged_response = tagged_response.strip().strip("\n")
    prompt = prompt_template.format(problem=data['problem'], tagged_response=tagged_response)
    new_data["prompt"] = prompt
    completion = ""
    for sdx in range(len(data['merged_critiques'])):
        completion = completion + data['merged_critiques'][sdx] + "\n\n"
    if -1 in data['critic_labels']:
        completion = completion + "**Answer**: \\boxed{{{}}}".format(str(len(data["merged_critiques"])))
    else:
        completion = completion + "**Answer**: \\boxed{{{}}}".format(str(-1))
    new_data["completion"] = completion
    response_len = len(tokenizer(completion)["input_ids"])
    
    if -1 in data['critic_labels']:
        label = len(data["merged_critiques"])
    else:
        label = -1
    if label not in label_len_dict:
        label_len_dict[label] = response_len
    else:
        label_len_dict[label] = max(label_len_dict[label], response_len)
    
    response_len_list.append(response_len)

    for i in range(len(data["merged_critiques"])):
        if data["critic_labels"][i] == -1:
            if data["critique_correctness"][i] == True:
                wrong_steps_no_correction += 1
            else:
                wrong_steps_has_correction += 1
        else:
            if data["critique_correctness"][i] == True:
                correct_steps_no_correction += 1
            else:
                correct_steps_has_correction += 1
    # Count label distribution
    if label not in label_distribution:
        label_distribution[label] = 1
    else:
        label_distribution[label] += 1

    new_data_list.append(new_data)

# check some statistics
print(len(new_data_list))
print(np.mean(response_len_list), np.max(response_len_list), np.min(response_len_list))
print(label_len_dict)
print(correct_steps_no_correction, correct_steps_has_correction, wrong_steps_no_correction, wrong_steps_has_correction)
print(label_distribution)

# process formatted data
output_file = "data/prm800k/phase2_train_final_critique_formatted.json"
original_data = new_data_list
new_data = []
for sample in original_data:
    # for SFT
    new_sample = {
        "messages": [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]}
        ],
    }
    
    new_data.append(new_sample)

with open(output_file, "w") as f:
    f.write(json.dumps(new_data))