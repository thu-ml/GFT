set -x
# Finetune
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12245 autoregressive/train/GFT_c2i_fsdp.py \
    --global-batch-size 256  --epochs=15 --ckpt-every=25000 --gradient-accumulation-steps=2 \
    --lr=2e-4 --expid="3843B_beta-1_lr24_ep15_maskbeta_detach_betainput2_refeval_cosinelr" --beta=-1.0 \
    --cosinelr=1 \
    --ref_ckpt="/data/home/chenhuayu/LlamaGen/c2i_3B_384.pt" \
    --code-path="/data/home/chenhuayu/imagenet384_train_code_c2i_flip_ten_crop/" \
    --results-dir="results" \
    --image-size=384 --gpt-model="GPT-3B"

# Train from scratch
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 autoregressive/train/GFT_c2i.py \
    --global-batch-size 768 --gradient-accumulation-step 2 --epochs=300 --ckpt-every=100000 \
    --lr=1e-4 --expid="384L_beta-1_lr14_bz768_ep1_maskbeta_detach_betainput2_fromscratch_cosinelr_fix" --beta=-1 --cosinelr=1 \
    --code-path="/data/home/chenhuayu/imagenet384_train_code_c2i_flip_ten_crop/" \
    --results-dir="results" \
    --image-size=384 --gpt-model="GPT-L"
