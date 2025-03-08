# Guidance Free Training script for LlamaGen

## Environment / Dataset Setup

Exactly the same as LlamaGen. See `LlamaGen_README.md`

## Model Evaluation

### Pretrained models
|   model    | reso. | #params |   FID  (w/o CFG)  | HF weights🤗                                                                        |
|:----------:|:-----:|:--------:|:---------:|:------------------------------------------------------------------------------------|
|  LlamaGen-L (pretrain)   |  384 |  343M   |   2.52    | [LlamaGenL_GF_scratch.pt](https://huggingface.co/ChenDRAG/LLamaGen_GF/blob/main/LlamaGenL_GF_scratch.pt) |
|  LlamaGen-3B (finetune)   |  384 |  3.0B   |   2.21    | [LlamaGen3B_GF.pt](https://huggingface.co/ChenDRAG/LLamaGen_GF/blob/main/LlamaGen3B_GF.pt) |

## Evaluation
Before evaluation, you should first generate 50K image samples and store them in an npz file.

```shell
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12445 LlamaGen_sample_ddp.py \
  --vq-ckpt="/data/home/chenhuayu/LlamaGen/vq_ds16_c2i.pt" \
  --gpt-ckpt="./LlamaGenL_GF_scratch.pt" \
  --gpt-model="GPT-L" --image-size=384 --image-size-eval=256 \
  --per-proc-batch-size=64 --cfg-scale=1.0 --beta=0.66 --num-fid-samples=50000 --sample-dir=samples
```

Please refer to `LlamaGen/sample.sh` for more usage reference.

We use the standard OPENAI evaluation metric to calculate FID and IS. Please refer to [./LlamaGen/evaluations/c2i](./LlamaGen/evaluations/c2i) for evaluation code.

## Model Training

After you have set up dataset according to LlamaGen instruction, run

```shell
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 autoregressive/train/GFT_c2i.py \
    --global-batch-size 768 --gradient-accumulation-step 2 --epochs=300 --ckpt-every=100000 \
    --lr=1e-4 --expid="384L_beta-1_lr14_bz768_ep1_maskbeta_detach_betainput2_fromscratch_cosinelr_fix" --beta=-1 --cosinelr=1 \
    --code-path="/data/home/chenhuayu/imagenet384_train_code_c2i_flip_ten_crop/" \
    --results-dir="results" \
    --image-size=384 --gpt-model="GPT-L"
```

Please refer to `LlamaGen/GFT.sh` for more usage reference.