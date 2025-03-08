torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12445 LlamaGen_sample_ddp.py \
  --vq-ckpt="/data/home/chenhuayu/LlamaGen/vq_ds16_c2i.pt" \
  --gpt-ckpt="/data/home/chenhuayu/git/LLamaGen_GF/LlamaGen3B_GF.pt" \
  --gpt-model="GPT-3B" --image-size=384 --image-size-eval=256 \
  --per-proc-batch-size=64 --cfg-scale=1.0 --beta=0.7 --num-fid-samples=50000 --sample-dir=samples2

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12445 LlamaGen_sample_ddp.py \
  --vq-ckpt="/data/home/chenhuayu/LlamaGen/vq_ds16_c2i.pt" \
  --gpt-ckpt="/data/home/chenhuayu/git/LLamaGen_GF/LlamaGenL_GF_scratch.pt" \
  --gpt-model="GPT-L" --image-size=384 --image-size-eval=256 \
  --per-proc-batch-size=64 --cfg-scale=1.0 --beta=0.66 --num-fid-samples=50000 --sample-dir=samples

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12445 LlamaGen_sample_ddp.py \
  --vq-ckpt="/data/home/chenhuayu/LlamaGen/vq_ds16_c2i.pt" \
  --gpt-ckpt="/data/home/chenhuayu/LlamaGen/c2i_L_384.pt" \
  --gpt-model="GPT-L" --image-size=384 --image-size-eval=256 \
  --per-proc-batch-size=64 --cfg-scale=2.0 --num-fid-samples=50000 --sample-dir=samples
