DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 accelerate launch \
    --config_file config/accelerate_config.yaml \
    ./finetune.py \