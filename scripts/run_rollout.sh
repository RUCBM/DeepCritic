# generate initial solutions
python3 Critique_Generation/gen_step_solutions.py \
    --input_file input_data_path \
    --model_path generator_path \
    --output_file output_data_path \
    --max_tokens 2048 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 64 \
    --n 8 

# filter out correct solutions and incorrect solutions separately for subsequent rollouts
python3 Critique_Generation/process_step_solutions.py

# rollout for correct solutions
python3 Critique_Generation/correct_solutions_rollout.py \
    --input_file input_data_path \
    --model_path generator_path \
    --output_file output_data_path \
    --max_tokens 4096 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_num_seqs 64 \
    --n 8

# rollout for incorrect solutions
python3 Critique_Generation/wrong_solutions_rollout.py \
    --input_file input_data_path \
    --model_path generator_path \
    --output_file output_data_path \
    --max_tokens 4096 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_num_seqs 64 \
    --n 8

# filter and curate final RL data