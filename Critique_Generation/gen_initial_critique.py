import re
import os
import sys
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
import copy
from vllm import LLM, SamplingParams


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, type=str, required=True)
parser.add_argument('--input_file', default=None, type=str, required=True)
parser.add_argument('--output_file', default=None, type=str, required=True)
parser.add_argument("--request_batch_size", type=int, default=1, help="Inference batch size.")
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.8, help="The temperature of generator.")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p.")
parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens.")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
parser.add_argument("--begin_idx", type=int, default=-1)
parser.add_argument("--end_idx", type=int, default=-1)


args = parser.parse_args()

def extract_completion_only(answer):
    # pattern = "<|start_header_id|>assistant<|end_header_id|>"
    pattern = ""
    results = []
    for an in answer:
        per_results = []
        for per_an in an.outputs:
            parts = per_an.text[len(pattern):].strip('\n')
            per_results.append(parts)
        results.append(per_results)
    return results

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    candidate = piece[:i]
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        candidate = piece[: i + 1]
                    
                    if is_int(candidate): 
                        answers.append(int(candidate))
                    break
    return answers

def extract_preds(outputs):
    preds = []
    for output in outputs:
        boxed_preds = extract_boxed_answers(output)
        if len(boxed_preds) > 0:
            pred = int(boxed_preds[-1])
        else:
            pred = None
        preds.append(pred)
    return preds


def preprocess_data(data_list):
    new_data_list = []
    used_problems = []
    for data in data_list:
        solutions = []
        problem = data["problem"]
        if problem in used_problems:
            continue
        steps = data["steps"]
        gt_answer = data["gt_answer"]
        contain_erroneous_step = data["contain_erroneous_step"]
        cur_solution = ""
        labels = []
        for i in range(len(steps)):
            step = steps[i]
            cur_solution = cur_solution + "Step {}: ".format(step["step_index"]) + step["text"] + "\n\n"
        cur_solution = cur_solution.strip("\n")
        for i in range(len(steps)):
            step = steps[i]
            if step["rating"] is not None:
                # cur_solution = cur_solution + "Step {}: ".format(step["step_index"]) + step["text"] + "\n\n"
                solutions.append(copy.deepcopy(cur_solution).strip("\n"))
                labels.append(step["rating"])
        new_data = {
            "problem": problem,
            "steps": steps,
            "critic_solutions": solutions,
            "critic_labels": labels,
            "gt_answer": gt_answer,
            "contain_erroneous_step": contain_erroneous_step,
        }
        new_data_list.append(new_data)
        used_problems.append(problem)

    return new_data_list

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    random.seed(args.seed)
    data_list = []
    with open(args.input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    random.shuffle(data_list)
    data_list = preprocess_data(data_list)
    if args.begin_idx != -1 and args.end_idx != -1:
        data_list = data_list[args.begin_idx: args.end_idx]
    print(f"Loaded {len(data_list)} samples")
    # print(data_list[200])

    file = open(os.path.join(script_dir, "critique_prompt_initial.txt"), "r")
    critique_prompt = file.read()
    file.close()
    prompt_template = (
                    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n{instruction}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    )
    llm = LLM(model=args.model, tensor_parallel_size=args.gpus)
    tokenizer = llm.get_tokenizer()

    prompt_samples = []
    for data_idx, data in enumerate(data_list):
        for i in range(len(data["critic_solutions"])):
            prompt_sample = [critique_prompt.format(problem=data["problem"], solution=data["critic_solutions"][i], step_index=(i + 1)), data["critic_labels"][i], data_idx, data]
            prompt_samples.append(prompt_sample)
    print(f"Processed {len(prompt_samples)} prompts")
    new_sample_dict = {}
    for i in tqdm(range(int(len(prompt_samples) // args.request_batch_size) + 1)):
        batch_prompt_samples = prompt_samples[i * args.request_batch_size : min(len(prompt_samples), (i + 1) * args.request_batch_size)]
        if len(batch_prompt_samples) == 0:
            break
        prompt_samples_for_model = [prompt_template.format(instruction=prompt_sample[0]) for prompt_sample in batch_prompt_samples]

        sampling_params = SamplingParams(temperature=args.temperature,
                                         top_p=args.top_p,
                                         max_tokens=args.max_tokens,
                                         seed=args.seed,
                                         n=args.n)
        outputs = llm.generate(prompt_samples_for_model, sampling_params)
        outputs = extract_completion_only(outputs)
        # print(outputs)
        pred_labels = []
        for outs in outputs:
            tmp_pred_labels = extract_preds(outs)
            pred_labels.append(tmp_pred_labels)
        for idx, outs in enumerate(outputs):
            # problem = batch_prompt_samples[idx][-1]["problem"]
            data_idx = batch_prompt_samples[idx][2]
            if data_idx not in new_sample_dict:
                new_sample_dict[data_idx] = batch_prompt_samples[idx][-1]
                new_sample_dict[data_idx]["critiques"] = []
                new_sample_dict[data_idx]["critique_correctness"] = []
            all_critiques, all_correct_critiques = [], []
            for j in range(len(outs)):
                pred_label = pred_labels[idx][j]
                if pred_label not in [-1, 1]:
                    continue
                all_critiques.append(outs[j])
                gt_label = batch_prompt_samples[idx][1]
                correctness = False
                if pred_label is not None:
                    correctness = (pred_label == 1) == (gt_label in [1, 0])
                if correctness:
                    all_correct_critiques.append(outs[j])
                
            if len(all_correct_critiques) > 0:
                selected_critique_index = 0
                new_sample_dict[data_idx]["critiques"].append(all_correct_critiques[selected_critique_index])
                new_sample_dict[data_idx]["critique_correctness"].append(True)
            elif len(all_critiques) > 0:
                selected_critique_index = 0
                new_sample_dict[data_idx]["critiques"].append(all_critiques[selected_critique_index])
                new_sample_dict[data_idx]["critique_correctness"].append(False)
            else:
                new_sample_dict[data_idx]["critiques"].append("")
                new_sample_dict[data_idx]["critique_correctness"].append(False)

    print("Writing results...")
    with open(args.output_file, 'w') as f:
        for data_idx, data in new_sample_dict.items():
            json_str = json.dumps(data)
            f.write(json_str + "\n")
    print("Done!")
    correct_solution_acc, wrong_solution_acc = [], []
    for data_idx, data in new_sample_dict.items():
        if data["contain_erroneous_step"]:
            if False in data["critique_correctness"]:
                wrong_solution_acc.append(0)
            else:
                wrong_solution_acc.append(1)
        else:
            if False in data["critique_correctness"]:
                correct_solution_acc.append(0)
            else:
                correct_solution_acc.append(1)

    print("Critic accuracy on correct solutions: ", np.mean(correct_solution_acc), "Number of correct solutions: ", len(correct_solution_acc))
    print("Critic accuracy on wrong solutions: ", np.mean(wrong_solution_acc), "Number of wrong solutions: ", len(wrong_solution_acc))


