set -x

# GFT Pretrain
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12245 train.py --model DiT-B/2 --results-dir workdir/DiT-B-Pretrain --data-path=/path/to/ILSVRC2012/train/ --rm-guidance --lr=1e-4 --epochs 80 --ckpt-every 100000 --beta_in=linear --cos-decay --pretrain

# GFT Finetune
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12245 train.py --model DiT-XL/2 --results-dir workdir/DiT-XL-Finetune --data-path=/path/to/ILSVRC2012/train/ --resume-ckpt=/path/to/DiT-XL-2-256x256.pt --rm-guidance --lr=1e-4 --epochs 28 --ckpt-every 10000 --beta_in=linear --cos-decay