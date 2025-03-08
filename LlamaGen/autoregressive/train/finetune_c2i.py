# Modified from ./LlamaGen/autoregressive/train/train_c2i.py

# # Include LlamaGen repo as a library
# import sys
# sys.path.append("./")

# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.expid}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
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

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    if args.cosinelr == 1:
        # Setup learning rate scheduler (Cosine Annealing)
        scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs * 5000) / (args.global_batch_size / 256), eta_min=1e-5)
        logger.info(f"Using cosine annealing LR scheduler with T_max={(args.epochs * 5000) / (args.global_batch_size / 256)}, eta_min={1e-5}.")
    
    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
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

    # Prepare models for training:
    if args.ref_ckpt:
        checkpoint = torch.load(args.ref_ckpt, map_location="cpu")
        weight = checkpoint["model"] if "XXL" not in args.ref_ckpt and "3B" not in args.ref_ckpt else checkpoint
        if "freqs_cis" in weight:
            weight.pop("freqs_cis")
        print(model.load_state_dict(weight, strict=False)) # TODO  here we should simply ingore incosistency head2
        if args.ema:
            ema.load_state_dict(weight)
        logger.info(f"Ref ckpt loaded.")
        del checkpoint
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        logger.info(f"Training from scratch.")
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
            
            
            
    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=False)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
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
            with torch.cuda.amp.autocast(dtype=ptdtype):
                if args.beta >= 0.0:
                    # _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
                    model.train()
                    cond_logits, cond_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
                    if args.beta == 1.0:
                        uncond_logits = torch.zeros_like(cond_logits)
                        uncond_loss = torch.zeros_like(cond_loss)
                        logits = cond_logits
                        loss=cond_loss
                    else:
                        uncond_indices = torch.ones_like(c_indices) * args.num_classes
                        with torch.no_grad():
                            uncond_logits, uncond_loss = model(cond_idx=uncond_indices, idx=z_indices[:,:-1], targets=z_indices)
                        logits = args.beta * cond_logits + (1-args.beta) * uncond_logits.detach()
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
                else:
                    betas = torch.rand(c_indices.shape[0]).to(device, non_blocking=True)
                    uncond_indices = torch.ones_like(c_indices) * args.num_classes
                     model.eval()
                    with torch.no_grad():
                        uncond_logits, uncond_loss = model(cond_idx=uncond_indices, idx=z_indices[:,:-1], targets=z_indices, betas=1.0, training_behavior=True)
                    model.train()
                    cond_logits, cond_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, betas=betas)
                    logits = betas[:,None,None] * cond_logits + (1-betas[:,None,None]) * uncond_logits.detach()
                    assert cond_logits.dim() == 3
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1),reduction="none")
                    loss = torch.mean(loss)

            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if (acc_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                acc_step = 0
            else:
                acc_step += 1
                continue 
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

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
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        # "optimizer": optimizer.state_dict(),
                        # "steps": train_steps,
                        # "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                dist.barrier()
            if args.cosinelr == 1:
                # Update the learning rate scheduler
                scheduler.step()
                logger.info(f"Epoch {epoch}: Adjusted learning rate to {scheduler.get_last_lr()}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1, help="GFT beta")
    parser.add_argument("--cosinelr", type=int, default=0, help="Use cosine learning rate scheduler if set to 1.")
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--expid", type=str, required=True, help='Identifier')
    parser.add_argument("--ref_ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
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
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)
