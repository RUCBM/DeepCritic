# initial critique generation
python3 Critique_Generation/gen_initial_critique.py \
    --model your_model_path \
    --input_file data/prm800k/phase2_train_processed.jsonl \
    --output_file data/prm800k/phase2_train_initial_critique.jsonl \
    --request_batch_size 512 \
    --n 1 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 1024 \
    --gpus 4 \
    --begin_idx 0 \
    --end_idx 10000

# in-depth critique generation
python3 Critique_Generation/gen_in_depth_critique.py \
    --model your_model_path \
    --input_file data/prm800k/phase2_train_initial_critique.jsonl \
    --output_file data/prm800k/phase2_train_in_depth_critique.jsonl \
    --request_batch_size 64 \
    --n 16 \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_tokens 1024 \
    --gpus 4 \
    --begin_idx 0 \
    --end_idx 10000


# final critique synthesis
python3 Critique_Generation/merge_critique.py \
    --input_file data/prm800k/phase2_train_in_depth_critique.jsonl \
    --output_file data/prm800k/phase2_train_final_critique.jsonl \
    --model your_model_path \
    --request_batch_size 128 \
    --n 1 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 2048 \
    --gpus 4 


# process raw critique data to sft data
python3 Critique_Generation/process_formatted_sft_data.py