import argparse
import os
import torch
import json
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from math_verify import parse, verify
import copy

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

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
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

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, n=args.n, seed=42)


    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = [json.loads(line) for line in file]
    if args.begin_idx >= 0 and args.end_idx >= 0:
        input_data = input_data[args.begin_idx: args.end_idx]
    chat_template = None
    if "qwen2.5" in args.model_path.lower():
        chat_template = (
                    "<|im_start|>system\nYou are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                    "<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\nStep 1:"
                    )
    elif "llama3.1" in args.model_path.lower():
        chat_template = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\nStep 1:"
            )
    prompt_token_ids = [apply_chat_template(toker, [{"role": "user", "content": e["prompt"]}], chat_template=chat_template)
                            for e in input_data]

    generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)


    res_data = []
    for i in range(len(input_data)):
        d = copy.deepcopy(input_data[i])
        gt_answer = input_data[i]["answer"]
        generated_solutions = ["Step 1: " + ee.text.strip() for ee in generations[i].outputs]
        tmp_answers, tmp_correctness = [], []
        for gen_solution in generated_solutions:
            pred_answer = extract_boxed_content(gen_solution)
            if pred_answer == "None":
                tmp_answers.append("")
                tmp_correctness.append(False)
            else:
                tmp_answers.append(pred_answer)
                tmp_correctness.append(verify(parse("\\boxed{" + gt_answer + "}"), parse("\\boxed{" + pred_answer + "}")))

        d["solutions"] = generated_solutions
        d["pred_answers"] = tmp_answers
        d["pred_correctness"] = tmp_correctness
        d["model"] = args.model_name
        valid = any(ans != "" and not correct for ans, correct in zip(tmp_answers, tmp_correctness)) and any(correctness for correctness in tmp_correctness)
        if valid:
            d["valid"] = True
        else:
            d["valid"] = False
 
 
        res_data.append(d)

    with open(args.output_file, "w", encoding="utf-8") as file:
        for d in res_data:
            file.write(json.dumps(d) + "\n")


if __name__ == '__main__':
    main()
