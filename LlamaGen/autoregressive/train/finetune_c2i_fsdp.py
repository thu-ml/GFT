# Modified from:
#   Large-DiT: https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py

#   Modified from train_c2i_fsdp.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import time
import inspect
import functools
import argparse
import contextlib
from glob import glob

from utils.logger import create_logger
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models



def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # auto_wrap_policy=size_based_auto_wrap_policy,
        # process_group=fs_init.get_data_parallel_group(),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    torch.cuda.synchronize()

    return model



def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.gpt_type == 'c2i', "FSDP only supports c2i currently."
    # =======================================
    #    Initialize Distributed Training
    # =======================================
    dist.init_process_group("nccl")
    # init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")
    

    # Setup an experiment folder:
    if global_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.expid}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        # cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{args.expid}-{model_string_name}/checkpoints"
        # os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        # logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)
    # training args
    logger.info(f"{args}")  



    # ======================================================
    #     Initialize model and resume
    # ======================================================
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        class_dropout_prob=0.1,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if global_rank == 0:  # other ranks receive weights in setup_fsdp_sync
        checkpoint = torch.load(args.ref_ckpt, map_location="cpu")
        weight = checkpoint["model"] if ("XXL" not in args.ref_ckpt and "3B" not in args.ref_ckpt) else checkpoint
        if "freqs_cis" in weight:
            weight.pop("freqs_cis")
        logger.info(f"Ref ckpt loaded.")
        train_steps = 0
        start_epoch = 0
    model = setup_fsdp_sync(model, args, device)


    # ======================================================
    #     Initialize optimizer and resume
    # ======================================================
    optimizer = creat_optimizer_by_name(model, args.weight_decay, args.lr, (args.beta1, args.beta2), global_rank, logger)
    if args.cosinelr == 1:
        # Setup learning rate scheduler (Cosine Annealing)
        # TODO here assume bz is 256
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * 5000, eta_min=1e-5)
        logger.info(f"Using cosine annealing LR scheduler with T_max={args.epochs}, eta_min={0}.")


    # ======================================================
    #     Initialize Dataloader
    # ======================================================
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int((args.global_batch_size // dist.get_world_size()) // args.gradient_accumulation_steps),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
    

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    train_steps = 0
    start_epoch = 0
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    cond_running_loss = 0
    uncond_running_loss = 0
    start_time = time.time()
    acc_step = 0

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            optimizer.zero_grad()
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]: 
                if args.beta >= 0.0:
                    # _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
                    cond_logits, cond_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
                    uncond_indices = torch.ones_like(c_indices) * args.num_classes
                    with torch.no_grad():
                        uncond_logits, uncond_loss = model(cond_idx=uncond_indices, idx=z_indices[:,:-1], targets=z_indices)
                    logits = args.beta * cond_logits + (1-args.beta) * uncond_logits.detach()
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
                else:
                    # betas = torch.rand(c_indices.shape[0]).to(device, non_blocking=True) * 0.9 + 0.1
                    betas = torch.rand(c_indices.shape[0]).to(device, non_blocking=True)
                    uncond_indices = torch.ones_like(c_indices) * args.num_classes
                    model.eval()
                    with torch.no_grad():
                        uncond_logits, uncond_loss = model(cond_idx=uncond_indices, idx=z_indices[:,:-1], targets=z_indices, betas=1.0, training_behavior=True)
                    model.train()
                    cond_logits, cond_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, betas=betas)
                    assert cond_logits.dim() == 3
                    logits = betas[:,None,None] * cond_logits + (1-betas[:,None,None]) * uncond_logits.detach()
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1),reduction="none")
                    loss = torch.mean(loss)
            loss.backward()
            # print("step {}".format(train_steps)+loss.item())
            if (acc_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                #   according to https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
                #   torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    model.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                acc_step = 0
            else:
                acc_step += 1
                continue
            

            # Log loss values:
            running_loss += loss.item()
            cond_running_loss += cond_loss.item()
            uncond_running_loss += uncond_loss.item()
            log_steps += 1
            train_steps += 1
            if (train_steps % args.log_every == 0) or (train_steps < 300):
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                cond_avg_loss = torch.tensor(cond_running_loss / log_steps, device=device)
                uncond_avg_loss = torch.tensor(uncond_running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(cond_avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(uncond_avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                cond_avg_loss = cond_avg_loss.item() / dist.get_world_size()
                uncond_avg_loss = uncond_avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Uncond Loss: {uncond_avg_loss:.4f}, Cond Loss: {cond_avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                cond_running_loss = 0
                uncond_running_loss = 0
                log_steps = 0
                start_time = time.time()
            # Save checkpoint:
            if (train_steps > 0) and ((train_steps % args.ckpt_every == 0) or (train_steps==5000)):
                ### saving model parameters
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if global_rank == 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(consolidated_model_state_dict, checkpoint_path)
                dist.barrier()
                del consolidated_model_state_dict
            if args.cosinelr == 1:
                # Update the learning rate scheduler
                scheduler.step()
                # logger.info(f"Epoch {epoch}: Adjusted learning rate to {scheduler.get_last_lr()}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1, help="GFT beta")
    parser.add_argument("--cosinelr", type=int, default=0, help="Use cosine learning rate scheduler if set to 1.")
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--expid", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--ref_ckpt", type=str, default=None, help="ckpt path for resume training")
    # parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-resume", type=str, default=None, help="model, optimizer and argument path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"], default='bf16') 
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--wandb-project", type=str, default='c2i_fsdp')
    parser.add_argument("--no-wandb", action='store_true')
    args = parser.parse_args()
    main(args)
