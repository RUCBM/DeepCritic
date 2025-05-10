for MODEL in your_model_path
do
python3 ProcessBench/code/eval.py \
    --task gsm8k \
    --input_file data/ProcessBench/gsm8k.json \
    --model_path ${MODEL} \
    --output_dir outputs \
    --use_voting

python3 ProcessBench/code/eval.py \
    --task math \
    --input_file data/ProcessBench/math.json \
    --model_path ${MODEL} \
    --output_dir outputs

python3 ProcessBench/code/eval.py \
    --task olympiadbench \
    --input_file data/ProcessBench/olympiadbench.json \
    --model_path ${MODEL} \
    --output_dir outputs


python3 ProcessBench/code/eval.py \
    --task omnimath \
    --input_file data/ProcessBench/omnimath.json \
    --model_path ${MODEL} \
    --output_dir outputs

python3 ProcessBench/code/eval.py \
    --task mrgsm8k \
    --input_file data/Mr-GSM8K/processed_test.json \
    --model_path ${MODEL} \
    --output_dir outputs


python3 ProcessBench/code/eval.py \
    --task prm800k \
    --input_file data/prm800k/phase2_test.json \
    --model_path ${MODEL} \
    --output_dir outputs

done