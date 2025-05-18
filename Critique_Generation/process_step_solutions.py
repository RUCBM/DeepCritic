
import json
import re
import os

def reformat_step_solution(solution):
    steps = re.findall(r'Step \d+:.*?(?=Step \d+:|$)', solution, re.DOTALL)

    if not steps:
        return None
    
    formatted_steps = []
    for i, step in enumerate(steps, 1):
        step = step.strip()
        
        if not step.startswith(f"Step {i}:"):
            step_match = re.match(r'Step (\d+):', step)
            if step_match:
                step = f"Step {i}:" + step[len(f"Step {step_match.group(1)}:"):]
            else:
                step = f"Step {i}: " + step
        
        formatted_steps.append(step)
    
    return formatted_steps

def process_solutions_files(input_files, output_file):
    all_processed_data = []
    total_correct_solutions = 0
    total_incorrect_solutions = 0
    
    for input_file in input_files:
        source = os.path.basename(input_file).split('_')[0]
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                if not data.get('valid', False):
                    continue
                
                correct_solutions = []
                incorrect_non_empty_solutions = []
                
                for i, is_correct in enumerate(data.get('pred_correctness', [])):
                    if is_correct:
                        if len(correct_solutions) < 1:
                            correct_solutions.append(data['solutions'][i])
                    elif (not is_correct and 
                          i < len(data.get('pred_answers', [])) and 
                          data['pred_answers'][i] != ""):
                        incorrect_non_empty_solutions.append(data['solutions'][i])
                
                total_correct_solutions += len(correct_solutions)
                total_incorrect_solutions += len(incorrect_non_empty_solutions)
                
                if correct_solutions or incorrect_non_empty_solutions:
                    reformatted_correct = [reformat_step_solution(sol) for sol in correct_solutions]
                    reformatted_incorrect = [reformat_step_solution(sol) for sol in incorrect_non_empty_solutions]
                    
                    new_data = {
                        'prompt': data.get('prompt', ''),
                        'answer': data.get('answer', ''),
                        'correct_solutions': reformatted_correct,
                        'incorrect_solutions': reformatted_incorrect,
                        'source': source
                    }
                    
                    all_processed_data.append(new_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processing complete, a total of {len(all_processed_data)} entries generated")
    print(f"Total number of correct solutions: {total_correct_solutions}")
    print(f"Total number of incorrect solutions: {total_incorrect_solutions}")
    print(f"Data saved to: {output_file}")

if __name__ == "__main__":
    input_files = [
        # input_file_paths
    ]
    output_file = "your_output_file_path"
    
    process_solutions_files(input_files, output_file)
