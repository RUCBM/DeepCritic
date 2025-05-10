import argparse
import numpy as np
import os
import torch
import json
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import re

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def apply_chat_template(toker, messages, chat_template=None):
    if chat_template is None:
        input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        input_prompt = chat_template.format(prompt=messages[0]["content"])
    return toker(input_prompt, add_special_tokens=False).input_ids

def prepare_input_boxed(template, input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        # tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
        if "Step " not in step:
            step = "Step {}: ".format((sdx + 1)) + step
        tagged_response = tagged_response + step + "\n\n"
    tagged_response = tagged_response.strip().strip("\n")
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    # messages = {"prompt": prompt}
    return messages

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--configs', type=str, nargs='+', default=None,
    #                     choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'])
    parser.add_argument('--task', type=str, default="gsm8k", required=True)
    parser.add_argument('--input_file', type=str, default="./data/ProcessBench/gsm8k.json", required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument('--use_voting', action='store_true')
    parser.add_argument('--voting_n', type=int, default=8)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    args.model_name = os.path.basename(args.model_path)

    toker = AutoTokenizer.from_pretrained(args.model_path)
    TEMPLATE = open(os.path.join(script_dir, 'templates/critique_template_new.txt')).read().strip()


    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True, swap_space=16,
        max_num_seqs=20, # can increase max_num_seqs with more GPUs
    )
    if not args.use_voting:
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=32768, seed=42)
    else:
        if 'Qwen2.5-Math' in args.model_path: # to ensure normal generation of Qwen2.5-Math-7B/72B-Instruct
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=args.voting_n,
                                            max_tokens=32768, seed=42)
        else:
            sampling_params = SamplingParams(temperature=1, top_p=0.9, n=args.voting_n,
                                            max_tokens=32768, seed=42)


    # if args.configs is None:
    #     args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    
    # for config in args.configs:
    if not args.use_voting:
        output_dir = os.path.join(args.output_dir, args.model_name)
    else:
        output_dir = os.path.join(args.output_dir, f'{args.model_name}_voting')
    os.makedirs(output_dir, exist_ok=True)

    # input_data = load_dataset(args.input_file)
    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = json.load(file)
    chat_template = None
    if "qwen2.5" in args.model_path.lower() and "math" not in args.model_path.lower():
        chat_template = (
                    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    )

    prompt_token_ids = [apply_chat_template(toker, prepare_input_boxed(TEMPLATE, e), chat_template=chat_template)
                            for e in input_data]

    generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)


    res_data = []
    for i in range(len(input_data)):
        d = input_data[i].copy()

        if not args.use_voting:
            generated_critique = generations[i].outputs[0].text
            pred = extract_answer(generated_critique)
            try:
                pred = int(pred)
            except:
                pred = None
        else:
            if "deepcritic" in args.model_path.lower(): # for DeepCritic models, we discard cases in which the final judgments are not provided
                generated_critique = []
                for ee in generations[i].outputs:
                    if "**Answer**:" in ee.text:
                        generated_critique.append(ee.text)
            else:
                generated_critique = [ee.text for ee in generations[i].outputs]
            preds = [extract_answer(e) for e in generated_critique]
            preds = [e for e in preds if e is not None]
            if len(preds) == 0:
                pred = None
            else:
                pred = Counter(preds).most_common(1)[0][0]
                try:
                    pred = int(pred)
                except:
                    pred = None

        d['generated_critique'] = generated_critique
        d['prediction'] = pred
        if d['label'] == -1:
            d['match'] = (pred == d['label'])
        else:
            d['match'] = (pred == (d['label'] + 1)) # step index begins from 1
        # d['match'] = (pred == d['label'])

        res_data.append(d)


    error_data = [e for e in res_data if e['label'] != -1]
    correct_data = [e for e in res_data if e['label'] == -1]

    with open(os.path.join(output_dir, f'{args.task}_error.jsonl'), 'w') as f:
        for e in error_data:
            f.write(json.dumps(e) + '\n')
    with open(os.path.join(output_dir, f'{args.task}_correct.jsonl'), 'w') as f:
        for e in correct_data:
            f.write(json.dumps(e) + '\n')
    
    acc1 = np.mean([e['match'] for e in error_data]) * 100
    acc2 = np.mean([e['match'] for e in correct_data]) * 100
    f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    print(f'{args.task} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')


if __name__ == '__main__':
    main()
