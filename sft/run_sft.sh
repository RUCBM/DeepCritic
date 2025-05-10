ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./sft/accelerate_configs/deepspeed_zero3.yaml \
    sft/run_sft.py \
    ./sft/model_config/config_sft.yaml

