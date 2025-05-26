from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler,  StableDiffusionPipeline, DDPMScheduler
from coco_eval.coco_evaluator import evaluate_model, compute_clip_score, compute_image_reward, CenterCropLongEdge
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate.utils import ProjectConfiguration
from utils import create_image_grid
from accelerate.logging import get_logger
from utils import SDTextDataset
from accelerate.utils import set_seed
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import argparse
import logging
import wandb
import torch
import glob
import time
import os
from PIL import Image
import clip
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from datetime import timedelta
from torchvision import transforms
import torch.distributed as dist

logger = get_logger(__name__, log_level="INFO")

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def resize_and_center_crop(image_np, resize_size=256):
    image_pil = Image.fromarray(image_np)
    image_pil = CenterCropLongEdge()(image_pil)

    if resize_size is not None:
        image_pil = image_pil.resize((resize_size, resize_size),
                                     Image.LANCZOS)
    return image_pil


def evaluate_aesthetic_score(image, model, model2, preprocess, device):
    image = resize_and_center_crop(image)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    prediction = model(torch.from_numpy(im_emb_arr).to(
        device).type(torch.cuda.FloatTensor))

    return prediction.item()


def evaluate_average_aesthetic_score(img_list, device):
    total_score = 0
    num_images = len(img_list)

    model = MLP(768)
    s = torch.load("sac+logos+ava1-l14-linearMSE.pth")
    model.load_state_dict(s)
    model.to("cuda")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load(
        "ViT-L/14", device=device)

    for img in tqdm(img_list):
        score = evaluate_aesthetic_score(
            img, model, model2, preprocess, device)
        total_score += score

    average_score = total_score / num_images if num_images > 0 else 0
    return average_score


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(w.device)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def get_x0_from_noise(sample, model_output, timestep):
    # alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    # 0.0047 corresponds to the alphas_cumprod of the last timestep (999)
    alpha_prod_t = (torch.ones_like(timestep).float()
                    * 0.0047).reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5)
                            * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


@torch.no_grad()
def sample(accelerator, dataloader, args,  pipeline):
    all_images = []
    all_captions = []
    all_index = []
    counter = 0

    set_seed(args.seed+accelerator.process_index)

    for index, batch_prompts in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process, total=args.total_eval_samples // args.eval_batch_size // accelerator.num_processes):
        # prepare generator input
        prompt_inputs = batch_prompts['key']

        generator = torch.Generator(device="cpu").manual_seed(index)

        eval_images = pipeline(
            prompt=prompt_inputs,
            guidance_scale=args.guidance_scale,
            output_type="np",
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images
        eval_images = (torch.tensor(eval_images, dtype=torch.float32)
                       * 255.0).to(torch.uint8).to(accelerator.device)

        gathered_images = accelerator.gather(eval_images)

        all_images.append(gathered_images.cpu().numpy())

        all_captions.append(batch_prompts['key'])

        gathered_index = accelerator.gather(
            batch_prompts['index'].to(accelerator.device))
        all_index += gathered_index

        counter += len(gathered_images)

        if counter >= args.total_eval_samples:
            break

    all_images = np.concatenate(all_images, axis=0)[:args.total_eval_samples]
    if accelerator.is_main_process:
        print("all_images len ", len(all_images))

    all_captions = [caption for sublist in all_captions for caption in sublist]
    data_dict = {"all_images": all_images,
                 "all_captions": all_captions, "all_index": all_index}

    accelerator.wait_for_everyone()
    return data_dict


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True,
                        help="pass to folder list")
    parser.add_argument("--clip_model", type=str, default="ViT-G/14")
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--anno_path", type=str)
    parser.add_argument("--eval_res", type=int, default=256)
    parser.add_argument("--ref_dir", type=str)
    parser.add_argument("--total_eval_samples", type=int, default=30000)
    parser.add_argument("--model_id", type=str,
                        default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--dpm_solver", action="store_true")
    parser.add_argument("--compute_aesthetic_score", action="store_true")
    parser.add_argument("--not_check_safety", action="store_true")
    args = parser.parse_args()

    args.folder = os.path.join(args.folder, os.path.basename(args.anno_path))
    folder = args.folder
    evaluated_checkpoints = set()
    overall_stats = {}

    timeout_long_ncll = timedelta(seconds=6000)  # 10 minutes
    dist.init_process_group(backend="nccl", timeout=timeout_long_ncll)

    os.makedirs(args.folder, exist_ok=True)

    # initialize accelerator
    accelerator_project_config = ProjectConfiguration(logging_dir=args.folder)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(f"folder to evaluate: {folder}", main_process_only=True)

    generator = None

    if args.not_check_safety:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
        )
    pipeline = pipeline.to(accelerator.device)

    if args.dpm_solver is True:
        # DPM Solver++
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config)

    if args.ckpt != "":
        unet = UNet2DConditionModel.from_pretrained(
            args.ckpt
        )
        pipeline.unet = unet.to(accelerator.device).float()

    pipeline.set_progress_bar_config(disable=True)
    pipeline.safety_checker = None

    # initialize tokenizer and dataset

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer"
    )
    caption_dataset = SDTextDataset(args.anno_path, tokenizer, is_sdxl=False)

    caption_dataloader = torch.utils.data.DataLoader(
        caption_dataset, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False, num_workers=8
    )
    caption_dataloader = accelerator.prepare(caption_dataloader)

    # generate images
    data_dict = sample(
        accelerator,
        caption_dataloader,
        args,
        pipeline=pipeline
    )
    torch.cuda.empty_cache()

    all_index = data_dict["all_index"]
    all_captions = [caption_dataset[i]['key'] for i in all_index]

    if accelerator.is_main_process:
        print("start fid eval")
        fid, prec, recall = evaluate_model(
            args, accelerator.device, data_dict["all_images"]
        )
        patch_fid = evaluate_model(
            args, accelerator.device, data_dict["all_images"], patch_fid=True
        )

        stats = {
            "fid": fid,
            "patch_fid": patch_fid,
            "prec": prec,
            "recall": recall
        }

        print(stats)
    
        clip_score = compute_clip_score(
            images=data_dict["all_images"],
            captions=all_captions,
            clip_model=args.clip_model,
            device=accelerator.device,
            how_many=args.total_eval_samples,
        )
        print(f"checkpoint {args.folder} clip score {clip_score}")
        stats['clip_score'] = float(clip_score)

    accelerator.wait_for_everyone()
    # in case of timeout

    if accelerator.is_main_process:
        if args.compute_aesthetic_score:
        aesthetic_score = evaluate_average_aesthetic_score(
            data_dict["all_images"], device=accelerator.device)
        print("aesthetic_score: ", aesthetic_score)
        stats['aesthetic_score'] = float(aesthetic_score)

        print(f"checkpoint {args.folder} fid {fid}")

        overall_stats[args.folder] = stats
        txt_path = os.path.join(args.folder, "results.txt")
        print("writing to {}".format(txt_path))
        with open(txt_path, 'a') as f:
            print("Clip Score:", stats['clip_score'], file=f)
            print("FID:", stats["fid"], file=f)
            print("Patch FID:", stats["patch_fid"], file=f)
            print("prec:", stats["prec"], file=f)
            print("recall:", stats["recall"], file=f)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    evaluate()
