__all__ = ["dpm_sample"]

from diffusers import DPMSolverMultistepScheduler
import torch
import einops


def diffusers_sample(nnet, sample_steps, z_init, device, model_kwargs, order=2):
    scheduler = DPMSolverMultistepScheduler(
        solver_order=order)
    scheduler.set_timesteps(sample_steps, device=device)
    timesteps = scheduler.timesteps

    latents = z_init
    for i, t in enumerate(timesteps):
        latents = scheduler.scale_model_input(latents, t)
        t_input = einops.repeat(t + 1, ' -> B', B=latents.size(0))
        noise_pred = nnet(latents, t_input, **model_kwargs)[:, :4]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


def sample(nnet, sample_steps, z_init, device, sampler, model_kwargs, t_start=None, t_end=None, solver=None, kwargs=None):
    if sampler == 'dpmsolver++':
        return diffusers_sample(nnet, sample_steps, z_init, device, model_kwargs)
    else:
        raise NotImplementedError
