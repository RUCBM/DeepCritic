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
from math_verify import parse, verify
import copy
import random

def apply_chat_template(toker, prompt, partial_solution, chat_template=None):
    """
    Apply chat template to messages.
    
    Args:
        toker: Tokenizer
        messages: List of message dictionaries
        chat_template: Optional template string
        
    Returns:
        Tokenized input IDs
    """
    if chat_template is None:
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": partial_solution}]
        input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        input_prompt = chat_template.format(prompt=prompt, partial_solution=partial_solution)
    return toker(input_prompt, add_special_tokens=False).input_ids

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    
    Args:
        text: Input text containing boxed content
        
    Returns:
        Extracted content from the last boxed expression
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def create_partial_solution(steps: list, n: int) -> str:
    """
    Create a partial solution by combining the first n steps.
    
    Args:
        steps: List of solution steps
        n: Number of steps to include
        
    Returns:
        Partial solution text
    """
    if n >= len(steps):
        return "\n\n".join(steps)
    return "\n\n".join(steps[:n])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="Input file containing solutions")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the LLM model")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save results")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument("--max_num_seqs", type=int, default=32, help="Maximum number of sequences")
    parser.add_argument("--n", type=int, default=5, help="Number of completions to generate per partial solution")
    parser.add_argument("--begin_idx", type=int, default=-1, help="Begin index for processing subset of data")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for processing subset of data")
    args = parser.parse_args()

    toker = AutoTokenizer.from_pretrained(args.model_path)
    args.model_name = os.path.basename(args.model_path)

    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True, swap_space=16,
        max_num_seqs=args.max_num_seqs,
    )
    random.seed(42)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, n=args.n, seed=42)

    # Load input data
    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = [json.loads(line) for line in file]
    
    filtered_data = input_data
        
    if args.begin_idx >= 0 and args.end_idx >= 0:
        filtered_data = filtered_data[args.begin_idx: args.end_idx]
    
    # Set up chat template based on model
    chat_template = None
    if "qwen2.5" in args.model_path.lower():
        chat_template = (
            "<|im_start|>system\nYou are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n{partial_solution}"
        )
    elif "llama3.1" in args.model_path.lower():
        chat_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{partial_solution}"
        )
    
    new_data_list = []
    
    # Prepare all prompts for batch processing
    all_prompts = []
    all_partial_solutions = []
    prompt_metadata = []  # Store metadata to map results back to data items
    
    for item_idx, item in enumerate(filtered_data):
        
        # Process incorrect solutions
        for solution_idx, step_list in enumerate(item["incorrect_solutions"]):
            wrong_steps = step_list
            # Process each partial solution (up to the second-to-last step)
            for step_idx in range(1, len(wrong_steps)):
                partial_solution = create_partial_solution(wrong_steps, step_idx)
                all_partial_solutions.append(partial_solution)
                prompt = item['prompt']
                all_prompts.append(prompt)
                prompt_metadata.append({
                    "item_idx": item_idx,
                    "solution_type": "wrong",
                    "solution_idx": solution_idx,
                    "step_idx": step_idx
                })
    
    # Tokenize all prompts
    prompt_token_ids = [apply_chat_template(toker, prompt, partial_solution, chat_template=chat_template)
                        for prompt, partial_solution in zip(all_prompts, all_partial_solutions)]
    
    # Generate completions for all prompts in batch
    generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    
    # Process results and organize by item
    results_by_item = {}
    
    for i, generation in enumerate(generations):
        meta = prompt_metadata[i]
        item_idx = meta["item_idx"]
        solution_type = meta["solution_type"]
        solution_idx = meta["solution_idx"]
        step_idx = meta["step_idx"]
        
        if item_idx not in results_by_item:
            results_by_item[item_idx] = {
                "wrong": {}
            }
        
        if solution_idx not in results_by_item[item_idx][solution_type]:
            results_by_item[item_idx][solution_type][solution_idx] = {}
        if step_idx not in results_by_item[item_idx][solution_type][solution_idx]:
            results_by_item[item_idx][solution_type][solution_idx][step_idx] = 0
        
        # Count correct completions
        gt_answer = filtered_data[item_idx]["answer"]
        
        for output in generation.outputs:
            completion = output.text
            pred_answer = extract_boxed_content(completion)
            if pred_answer != "None":
                is_correct = verify(parse("\\boxed{" + gt_answer + "}"), parse("\\boxed{" + pred_answer + "}"))
                if is_correct:
                    results_by_item[item_idx][solution_type][solution_idx][step_idx] += 1

    
    # Create final output data
    for item_idx, item in enumerate(filtered_data):
        d = copy.deepcopy(item)
        
        # Extract results for this item
        item_results = results_by_item.get(item_idx, {"wrong": {}})
        
        # Organize wrong solution results for each incorrect solution
        wrong_partial_rollout_correct_numbers = []
        
        for solution_idx, step_list in enumerate(item["incorrect_solutions"]):
            solution_results = []
            
            for step_idx in range(1, len(step_list)):
                correct_count = item_results["wrong"].get(solution_idx, {}).get(step_idx, 0)
                solution_results.append(correct_count)
            
            wrong_partial_rollout_correct_numbers.append(solution_results)
        
        # Add new fields to the data
        d["wrong_solution_rollout_correct_numbers"] = wrong_partial_rollout_correct_numbers
        d["rollout_model"] = args.model_name
        
        new_data_list.append(d)

    # Save results to output file
    with open(args.output_file, "w", encoding="utf-8") as file:
        for d in new_data_list:
            file.write(json.dumps(d) + "\n")
    
    print(f"Processed {len(new_data_list)} items and saved to {args.output_file}")

if __name__ == '__main__':
    main()
