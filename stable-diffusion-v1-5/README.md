# Guidance Free Training script for Stable Diffusion 1.5

## Environment / Dataset Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 📂 Evaluation Dataset: COCO2014
The **COCO2014** dataset is used as the evaluation benchmark. To prepare it, please download the images and corresponding test captions using the following commands:
```bash
wget https://huggingface.co/tianweiy/DMD2/resolve/main/data/coco/captions_coco14_test.pkl?download=true -O coco/captions_coco14_test.pkl
wget https://huggingface.co/tianweiy/DMD2/resolve/main/data/coco/val2014.zip?download=true -O coco/val2014.zip
unzip coco/val2014.zip -d coco
```

### 📂 Training Dataset: LAION Aesthetics 5+
A subset of the **LAION Aesthetics 5+** dataset is used for training. To prepare the dataset, you need to create a `jsonl` file containing the image entries with the following format:
```json
{"caption": "Antique Khotan, East Turkestan, late 19th century, 3 feet 6 inches x 5 feet 3 inches. Estimate: $2,000-$4,000. Image courtesy of Nazmiyal Collection.", "image_path": "/path/to/image/0000295806.jpg"}
{"caption": "7 Cookbooks by Black Chefs That Serve Up More Than Just Meals", "image_path": "/path/to/image//0000291187.jpg"}
{"caption": "Circular Oak Occasional Table", "image_path": "/path/to/image/0000295233.jpg"}

```
Once the `jsonl` file is ready, you can pass its path to the training script using the `--train_data_jsonl` argument. You can refer to `train.sh` for a complete example of how to launch training with all necessary arguments.

## Model Evaluation

### Pretrained models
|              model              | reso. | FID  (w/o CFG) | HF weights🤗                                                                           |
| :-----------------------------: | :---: | :------------: | :------------------------------------------------------------------------------------ |
| stable diffusion 1.5 (finetune) |  512  |      8.10      | [SD1.5-GF-finetune](https://huggingface.co/aaa-ceku7/GFT/tree/main/SD1.5-GF-finetune) |

### Evaluation
Please refer to `stable-diffusion-v1-5/eval.sh` for usage reference.

## Model Training
Please refer to `stable-diffusion-v1-5/train.sh` for usage reference.

## Acknowledgements
This codebase includes components adapted from the following repositories:

- [Hugging Face diffusers](https://github.com/huggingface/diffusers), specifically the training script [`train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).
- [Tianwei Yin's DMD2](https://github.com/tianweiy/DMD2), particularly the evaluation script [`test_folder_sd.py`](https://github.com/tianweiy/DMD2/blob/main/main/test_folder_sd.py).

We thank the authors for their clear and well-structured implementations.
