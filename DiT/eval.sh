set -x

ckpt=DiT-XL-Finetune
step=0140000
sampling_steps=50

# GFT Sampling
for cfg_scale in 1.0 1.25 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.75 2.0 3.0 5.0; do
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12245 \
    sample_ddp.py \
    --vae-path=/data/public/sd-vae-ft-ema \
    --model=DiT-XL/2 \
    --ckpt=workdir/${ckpt}/000-DiT-XL-2/checkpoints/${step}.pt \
    --sample-dir=${ckpt}/${sampling_steps}/ \
    --dpm-solver \
    --cfg-scale=${cfg_scale} \
    --beta-in=linear \
    --rm-guidance \
    --num-sampling-steps ${sampling_steps}
    python evaluation.py \
    --sample_batch=${ckpt}/${sampling_steps}/${step}-size-256-cfg-${cfg_scale}-seed-0.npz
done 


