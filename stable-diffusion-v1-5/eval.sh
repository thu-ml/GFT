set -x
export PYTHONPATH=$PYTHONPATH:stable-diffusion-v1-5/coco_eval

model="/path/to/checkpoints"
for cfg_scale in 1.0 2.0 2.5 3.0 3.5 4.0 5.0 7.5 10.0 12.5 15.0; do
    accelerate launch --mixed_precision="fp16" test.py \
        --folder results/${model}/${cfg_scale} \
        --seed 10 \
        --anno_path coco/captions_coco14_test.pkl \
        --ref_dir coco/val2014 \
        --total_eval_samples 30000 \
        --model_id "sd-legacy/stable-diffusion-v1-5" \
        --guidance_scale ${cfg_scale} \
        --ckpt ${model} \
        --dpm_solver \
        --not_check_safety
done