# Guidance Free Training script for DiT

## Environment / Dataset Setup
Exactly the same as DiT. See `DiT_README.md`

## Model Evaluation

### Pretrained models
|        model        | reso. | FID  (w/o CFG) | HF weights🤗                                                                                   |
| :-----------------: | :---: | :------------: | :-------------------------------------------------------------------------------------------- |
| DiT-B/2 (pretrain)  |  256  |      9.04      | [DiT-B-GF-scratch.pt](https://huggingface.co/aaa-ceku7/GFT/blob/main/DiT-B-GF-scratch.pt)     |
| DiT-XL/2 (finetune) |  256  |      1.99      | [DiT-XL-GF-finetune.pt](https://huggingface.co/aaa-ceku7/GFT/blob/main/DiT-XL-GF-finetune.pt) |

### Evaluation
Please refer to `DiT/eval.sh` for usage reference.

## Model Training
Please refer to `DiT/train.sh` for usage reference.

## Acknowledgements
This codebase is partially adapted from the [`DiT`](https://github.com/facebookresearch/DiT). We thank the authors for their clean and well-documented implementation.