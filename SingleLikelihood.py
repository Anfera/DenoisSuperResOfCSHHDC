import gc
import os
import warnings

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPSILON,
    FACTOR,
    IMAGE_SIZE,
    INTERMEDIATE_DIR,
    LR_MULTIPLIER,
    MASK_RATIO,
    MASK_TYPE,
    PHOTONS,
    RECON_THRESHOLD,
    RESOLUTION,
    SAMPLING_TIMESTEPS,
    SEED,
    SNR_DB,
    RESULT_DIR,
    checkpoint_path,
    seed_everything,
)
from src.data_utils import load_test_data
from src.forwardImagingPoisson import ForwardImaging
from src.mask_utils import create_mask
from src.visualization import plot_results
from src.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

warnings.filterwarnings("ignore")

ETA = 1.0


def prepare_output_dirs() -> None:
    """Ensure expected output folders exist."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)


def build_time_pairs(total_timesteps: int) -> list[tuple[int, int]]:
    """Construct reverse-ordered timestep pairs for DDIM sampling."""
    times = torch.linspace(-1, 1000 - 1, steps=total_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    return list(zip(times[:-1], times[1:]))


def configure_diffusion_model() -> tuple[Trainer, callable]:
    """Instantiate diffusion components and load the EMA weights."""
    model = Unet(dim=128, dim_mults=(8, 16, 16, 16), flash_attn=True, channels=128)

    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        timesteps=1000,
        sampling_timesteps=SAMPLING_TIMESTEPS,
    )

    trainer = Trainer(
        diffusion,
        train_batch_size=8,
        train_lr=8e-5,
        train_num_steps=700000,
        gradient_accumulate_every=8,
        ema_decay=0.995,
        amp=True,
        resolution=RESOLUTION,
        inference_only=True,
        results_folder=os.path.dirname(checkpoint_path()) or ".",
    )

    ckpt = checkpoint_path()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt}.")

    trainer.load(0)
    trainer.ema.ema_model.eval()

    def unet_wrapper(x, time_cond):
        return trainer.ema.ema_model.model_predictions(
            x, time_cond, x_self_cond=None, clip_x_start=True, rederive_pred_noise=True
        )

    return trainer, unet_wrapper


def add_noise(input_data: torch.Tensor, target_snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise to match the target SNR."""
    rms = torch.sqrt((input_data.abs() ** 2).mean())
    noise_level = rms / (10 ** (target_snr_db / 20))
    noisy = input_data + noise_level * torch.randn_like(input_data)
    print(f"Using noise level: {noise_level.item():.4f}")
    return noisy, noise_level


def prepare_mask(y: torch.Tensor) -> torch.Tensor:
    """Create and broadcast the sampling mask."""
    mask = create_mask(y.shape, ratio=MASK_RATIO, mask_type=MASK_TYPE, device=DEVICE)
    return mask.expand_as(y).float()


def main():
    seed_everything(SEED)
    prepare_output_dirs()

    forward_imaging = ForwardImaging(RESOLUTION, device=DEVICE, photons=PHOTONS)

    input_data, ground_truth = load_test_data(resolution=RESOLUTION, factor=FACTOR, device=DEVICE)
    input_data = forward_imaging.forward_imaging(ground_truth.unsqueeze(0).float())[0].sample()

    trainer, unet_wrapper = configure_diffusion_model()
    time_pairs = build_time_pairs(SAMPLING_TIMESTEPS)

    y, noise_level = add_noise(input_data, target_snr_db=SNR_DB)
    mask = prepare_mask(y)

    output = torch.randn_like(ground_truth.unsqueeze(0).float(), device=DEVICE)

    pbar = tqdm(time_pairs, total=SAMPLING_TIMESTEPS)
    for time, time_next in pbar:
        output = output.detach().requires_grad_(True)
        time_cond = torch.full((1,), time, device=DEVICE, dtype=torch.long)
        pred_noise, x_start, *_ = checkpoint(unet_wrapper, output, time_cond, use_reentrant=False)

        if time_next < 0:
            output = x_start
            continue

        alpha = trainer.model.alphas_cumprod[time]
        alpha_next = trainer.model.alphas_cumprod[time_next]

        sigma = ETA * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma**2).sqrt()

        with torch.no_grad():
            noise = torch.randn_like(output)
            output_p = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        x_start = ((x_start + 1) * 0.5) + EPSILON

        np.savez_compressed(
            os.path.join(INTERMEDIATE_DIR, f"x_start_{time}.npz"),
            reconstruction=x_start[0].detach().cpu().numpy(),
        )

        distri, x_aggregated = forward_imaging.forward_imaging(x_start, mask)
        mu = forward_imaging.photons * x_aggregated
        var_min = 0.05
        var = (mu + noise_level**2).clamp(min=var_min)
        res = y - mu

        nll = 0.5 * ((res**2) / var + torch.log(var))
        nll = (mask * nll).sum()

        grads = torch.autograd.grad(nll, output, create_graph=False)[0]

        sigma_scale = (noise_level.detach().float() + 1e-3).item()
        lr = LR_MULTIPLIER * (time / SAMPLING_TIMESTEPS)
        lr = lr * sigma_scale
        output = output_p - lr * grads

        with torch.no_grad():
            x_start[x_start < RECON_THRESHOLD] = 0
            norm = (x_start - ground_truth.unsqueeze(0)).abs().mean()

        pbar.set_postfix(
            norm=norm.item(),
            lr=lr,
            log_likelihood=nll.item() / (mask.sum() + 1e-6),
            time=time,
        )

        del nll, grads, output_p, norm, distri, x_aggregated, mu, var, res, x_start, pred_noise

        if time % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[t={time}] allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

    output = ((output.detach() + 1) * 0.5)

    recon = output[0].cpu().numpy()
    recon[recon < RECON_THRESHOLD] = 0

    np.save(os.path.join(RESULT_DIR, "mask.npy"), mask.cpu().numpy())
    np.save(os.path.join(RESULT_DIR, "input_data.npy"), y.cpu().numpy())

    contorno = (ground_truth.sum(0) > 0).float().cpu().numpy()
    normalized_input = y / (y.max(0)[0] + EPSILON)
    plot_results(normalized_input * mask, recon * contorno, ground_truth)


if __name__ == "__main__":
    main()
