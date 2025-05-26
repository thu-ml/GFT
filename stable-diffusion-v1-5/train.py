import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate.logging import get_logger
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import json
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import logging
from glob import glob
import wandb
from functools import partial


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


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(
                f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--rm_guidance", default=False, action="store_true",
    )
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.1)
    parser.add_argument("--our_beta", default=False, action="store_true")
    parser.add_argument("--distill", default=False, action="store_true")
    parser.add_argument("--min_omega", type=float, default=0.0)
    parser.add_argument("--max_omega", type=float, default=1.0)
    parser.add_argument("--pho", type=float, default=1.0)
    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=512,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_jsonl",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true",
                        help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true",
                        help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true",
                        help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.train_data_jsonl is None:
        raise ValueError("Need a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    dist.init_process_group("nccl")
    # Make one log on every process with the configuration for debugging.
    global_batch_size = dist.get_world_size() * args.train_batch_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    global_seed = args.seed
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        # Make results folder (holds all experiment subfolders)
        os.makedirs(args.output_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.output_dir}/*"))
        # Create an experiment folder
        experiment_dir = f"{args.output_dir}"
        # Stores saved model checkpoints
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(dir=os.path.abspath(experiment_dir), config=vars(args),
                   name=args.output_dir, job_type='train', resume=True)
    else:
        logger = create_logger(None)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    if args.distill:
        assert args.cond_dropout_prob == 0.0
        teacher_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )
        teacher_unet.eval()
        teacher_unet.requires_grad_(False)

    if args.rm_guidance:
        config = UNet2DConditionModel.load_config(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision
        )
        unet = UNet2DConditionModel.from_config(
            config, time_cond_proj_dim=args.unet_time_cond_proj_dim)
        unet.load_state_dict(
            UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                revision=args.non_ema_revision
            ).state_dict(),
            strict=False
        )
        torch.nn.init.normal(
            unet.time_embedding.cond_proj.weight, std=0)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )

    unet = DDP(unet.to(device), device_ids=[device])

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_config(unet.config)
        ema_unet.load_state_dict(unet.state_dict())
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
            decay=args.ema_decay,
        )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer:
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps
    num_training_steps_for_scheduler = args.max_train_steps

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(device)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        scaler = torch.amp.GradScaler()
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if args.resume_path is not None:
        filename = os.path.basename(args.resume_path)
        checkpoint_number = filename.split('-')[-1]
        global_step = initial_global_step = int(checkpoint_number)
        unet.module.load_state_dict(
            UNet2DConditionModel.from_pretrained(
                os.path.join(args.resume_path, "unet"),
                subfolder="unet",
                revision=args.non_ema_revision
            ).state_dict(),
            True,
        )
        ema_unet.load_state_dict(
            UNet2DConditionModel.from_pretrained(
                os.path.join(args.resume_path, "ema_unet"),
                subfolder="unet",
                revision=args.non_ema_revision
            ).state_dict(),
        )
        optimizer.load_state_dict(torch.load(
            os.path.join(args.resume_path, "optimizer"),  map_location='cpu'))
        lr_scheduler.load_state_dict(torch.load(
            os.path.join(args.resume_path, "lr_scheduler")))
        if args.mixed_precision == "fp16":
            scaler.load_state_dict(torch.load(
                os.path.join(args.resume_path, "scaler")))

    unet.train()
    image_column = args.image_column
    caption_column = args.caption_column

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []

        for example in examples:
            caption = example[caption_column]
            if isinstance(caption, str):
                captions.append(caption)
            else:
                raise NotImplementedError

        dropout_prob = args.cond_dropout_prob
        drop_ids = torch.rand(len(captions)) < dropout_prob
        captions = ["" if drop else caption for caption,
                    drop in zip(captions, drop_ids)]

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids, drop_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class JsonlImageCaptionDataset(Dataset):
        def __init__(self, jsonl_file, transform=None):
            """
            Args:
                jsonl_file (str): Path to the .jsonl file, where each line is a JSON object
                                containing at least 'filename' and 'caption' fields.
                transform (callable, optional): Optional transform to be applied on the image.
            """
            self.transform = transform
            self.samples = []

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                self.samples = f.readlines()

            if args.max_train_samples is not None:
                self.samples = self.samples[:args.max_train_samples]
                print("max_train_samples: ", args.max_train_samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = json.loads(self.samples[idx].strip())
            filename = sample[image_column]
            caption = sample[caption_column]

            filename = os.path.join(filename)

            # Load the image
            try:
                image = Image.open(filename).convert("RGB")
            except:
                alt_image_idx = (idx + 10000) % 1_000_000
                print(f"fail to load image {idx}, load image {alt_image_idx}")
                return self.__getitem__(alt_image_idx)

            # Apply transform if provided
            if self.transform:
                image = self.transform(image)

            return {
                "pixel_values": image,
                caption_column: caption
            }

    train_dataset = JsonlImageCaptionDataset(jsonl_file=args.train_data_jsonl, transform=train_transforms)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=global_seed
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"]
                                   for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        input_ids, drop_ids = tokenize_captions(examples)
        # input_ids = torch.stack(input_ids)
        return {"pixel_values": pixel_values.to(device), "input_ids": input_ids.to(device), "drop_ids": drop_ids.to(device)}

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    if args.distill:
        teacher_unet.to(device, dtype=weight_dtype)

    ctx = partial(torch.amp.autocast, "cuda", dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    if args.rm_guidance:
        uncond_input = tokenizer(
            [""] * args.train_batch_size, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        uncond_embeds = text_encoder(uncond_input.to(
            device), return_dict=False)[0]

        if not args.distill:
            if args.our_beta:
                w_uncond = get_timestep_embedding(
                    torch.full((args.train_batch_size, ), 1000.0).to(
                        device), embedding_dim=args.unet_time_cond_proj_dim).to(weight_dtype)
            else:
                w_uncond = guidance_scale_embedding(torch.full((args.train_batch_size, ), 0.0).to(
                    device), embedding_dim=args.unet_time_cond_proj_dim).to(weight_dtype)

        # Train!
    total_batch_size = args.train_batch_size * \
        dist.get_world_size() * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(args)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=rank != 0,
    )

    unet.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        avg_loss = 0
        if args.rm_guidance:
            loss_uncond = 0.0
            loss_psample = 0.0
            avg_loss_uncond = 0.0
            avg_loss_psample = 0.0
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            for _ in range(args.gradient_accumulation_steps):
                with ctx():
                    latents = vae.encode(batch["pixel_values"].to(
                        weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps)
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"], return_dict=False)[0]

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(
                            prediction_type=args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        raise NotImplementedError

                    # Predict the noise residual and compute loss
                    if args.rm_guidance:
                        beta = torch.rand(args.train_batch_size).to(
                            device) ** args.pho * (args.max_omega - args.min_omega) + args.min_omega

                        beta_in = torch.where(batch["drop_ids"], 1.0, beta)
                        if args.cond_dropout_prob == 0.0:
                            assert torch.equal(beta_in, beta)

                        if args.our_beta:
                            w_embedding = get_timestep_embedding(
                                beta_in * 1000, embedding_dim=args.unet_time_cond_proj_dim).to(weight_dtype)
                        elif args.distill:
                            w_embedding = guidance_scale_embedding(
                                beta_in - 1, embedding_dim=args.unet_time_cond_proj_dim).to(weight_dtype)
                        else:
                            w_embedding = guidance_scale_embedding(
                                torch.reciprocal(beta_in) - 1, embedding_dim=args.unet_time_cond_proj_dim).to(weight_dtype)

                        model_pred = unet(noisy_latents, timesteps, timestep_cond=w_embedding,
                                          encoder_hidden_states=encoder_hidden_states, return_dict=False)[0].float()
                        unet.eval()
                        with torch.no_grad():
                            if args.distill:
                                model_pred_uncond = teacher_unet(noisy_latents, timesteps,
                                                                 encoder_hidden_states=uncond_embeds.clone(), return_dict=False)[0].float().detach()
                            else:
                                model_pred_uncond = unet(noisy_latents, timesteps,
                                                         encoder_hidden_states=uncond_embeds.clone(), return_dict=False)[0].float().detach()
                            if args.distill:
                                target = teacher_unet(noisy_latents, timesteps,
                                                      encoder_hidden_states=encoder_hidden_states, return_dict=False)[0].float().detach()

                        unet.train()
                        beta = beta.reshape(args.train_batch_size, 1, 1, 1)
                        if args.distill and not args.our_beta:
                            target = model_pred_uncond + beta * \
                                (target - model_pred_uncond)
                            interpolated_output = model_pred
                        else:
                            interpolated_output = beta * model_pred + \
                                (1 - beta) * model_pred_uncond.detach()
                        loss = F.mse_loss(interpolated_output,
                                          target.float(), reduction="mean")
                        uncond_loss = F.mse_loss(model_pred_uncond.detach(),
                                                 noise.float(), reduction="mean")
                        psample_loss = F.mse_loss(model_pred.detach(),
                                                  noise.float(), reduction="mean")
                    else:
                        model_pred = unet(noisy_latents, timesteps,
                                          encoder_hidden_states, return_dict=False)[0]
                        loss = F.mse_loss(model_pred.float(),
                                          target.float(), reduction="mean")

                loss /= args.gradient_accumulation_steps
                if args.rm_guidance:
                    uncond_loss /= args.gradient_accumulation_steps
                    psample_loss /= args.gradient_accumulation_steps

                if args.mixed_precision == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                avg_loss += loss
                if args.rm_guidance:
                    avg_loss_uncond += uncond_loss
                    avg_loss_psample += psample_loss

            if args.mixed_precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            train_loss += avg_loss.item() / \
                dist.get_world_size()

            if args.rm_guidance:
                dist.all_reduce(avg_loss_uncond, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_psample, op=dist.ReduceOp.SUM)
                loss_uncond += avg_loss_uncond.item() / \
                    dist.get_world_size()
                loss_psample += avg_loss_psample.item() / \
                    dist.get_world_size()

            ema_unet.step(unet.parameters())
            progress_bar.update(1)
            global_step += 1
            log_dict = {"train_loss": train_loss, "step": global_step}
            if args.rm_guidance:
                log_dict.update(
                    {"loss_uncond": loss_uncond, "loss_psample": loss_psample})
            if rank == 0:
                logger.info(log_dict)
                wandb.log(log_dict)

            if global_step % args.checkpointing_steps == 0:
                if rank == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    unet.module.save_pretrained(
                        os.path.join(save_path, "unet"))
                    torch.save(optimizer.state_dict(),
                               os.path.join(save_path, "optimizer"))
                    torch.save(lr_scheduler.state_dict(),
                               os.path.join(save_path, "lr_scheduler"))
                    torch.save(global_step, os.path.join(
                        save_path, 'step.pth'))
                    if args.mixed_precision == "fp16":
                        torch.save(scaler.state_dict(),
                                   os.path.join(save_path, "scaler"))
                    if args.use_ema:
                        ema_unet.save_pretrained(
                            os.path.join(save_path, "ema_unet"))
                    logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": train_loss, "lr": lr_scheduler.get_last_lr()[
                0]}
            progress_bar.set_postfix(**logs)

            train_loss = 0.0
            avg_loss = 0.0

            loss_uncond = 0.0
            loss_psample = 0.0
            avg_loss_uncond = 0.0
            avg_loss_psample = 0.0

            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    main()
