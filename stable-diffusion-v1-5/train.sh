set -x

MODEL_NAME="sd-legacy/stable-diffusion-v1-5"

# GFT Finetune
exp=gft
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12245 train.py \
  --mixed_precision "fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_jsonl /path/to/laion_aesthetics_5plus.jsonl \
  --image_column="image_path" \
  --caption_column="caption" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=500000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=1000 \
  --output_dir="workdir/${exp}" \
  --rm_guidance \
  --logging_dir="workdir/${exp}" \
  --cond_dropout_prob 0.0 \
  --ema_decay 0.9999 \
  --seed 0 \
  --checkpointing_steps 10000