import gc
import os
import warnings

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from src.config import (
    BACKGROUND_RATE,
    DEVICE,
    EPSILON,
    FACTOR,
    FOOTPRINT_DIAMETER_M,
    IMAGE_SIZE,
    INPUT_RES_M,
    INTERMEDIATE_DIR,
    LR_MULTIPLIER,
    MASK_RATIO,
    MASK_TYPE,
    OUTPUT_RES_M,
    READOUT_NOISE,
    RECON_THRESHOLD,
    REF_ALTITUDE,
    REF_PHOTON_COUNT,
    RESOLUTION,
    RESULT_DIR,
    SAMPLING_TIMESTEPS,
    SEED,
    checkpoint_path,
    seed_everything,
)
from src.data_utils import load_test_data
from src.forward_model import LidarForwardImagingModel
from src.mask_utils import create_mask
from src.visualization import plot_results
from src.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

warnings.filterwarnings("ignore")

# DDIM stochasticity: 1.0 = full DDPM-equivalent noise at each step.
DDIM_ETA = 1.0


def prepare_output_dirs() -> None:
    """Ensure expected output folders exist."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)


def build_time_pairs(total_timesteps: int) -> list[tuple[int, int]]:
    """Construct reverse-ordered timestep pairs for DDIM sampling."""
    times = torch.linspace(-1, 1000 - 1, steps=total_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    return list(zip(times[:-1], times[1:]))


def configure_diffusion_model() -> tuple[object, callable]:
    """Instantiate diffusion components and load the EMA checkpoint."""
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


def main():
    seed_everything(SEED)
    prepare_output_dirs()

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    forward_imaging = LidarForwardImagingModel(
        input_res_m=INPUT_RES_M,
        output_res_m=OUTPUT_RES_M,
        footprint_diameter_m=FOOTPRINT_DIAMETER_M,
        b=BACKGROUND_RATE,
        eta=READOUT_NOISE,
        ref_altitude=REF_ALTITUDE,
        ref_photon_count=REF_PHOTON_COUNT,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ground_truth = load_test_data(resolution=RESOLUTION, factor=FACTOR, device=DEVICE)

    with torch.no_grad():
        y, _ = forward_imaging(ground_truth.float())
    # y shape: (num_bins, out_h, out_w)  — squeeze applied by forward model for 3-D input

    # ------------------------------------------------------------------
    # Sampling mask  — 2D spatial mask that broadcasts over the depth axis
    # ------------------------------------------------------------------
    mask = create_mask(
        y.shape, ratio=MASK_RATIO, mask_type=MASK_TYPE, device=DEVICE
    ).float()  # shape: (out_h, out_w)

    print(f"Observations: {tuple(y.shape)} | Mask: {tuple(mask.shape)} "
          f"({mask.mean().item() * 100:.1f} % sampled)")

    # ------------------------------------------------------------------
    # Diffusion model
    # ------------------------------------------------------------------
    trainer, unet_wrapper = configure_diffusion_model()
    time_pairs = build_time_pairs(SAMPLING_TIMESTEPS)

    # ------------------------------------------------------------------
    # Guided DDIM loop  (DPS: Chung et al. 2022)
    # ------------------------------------------------------------------
    output = torch.randn_like(ground_truth.unsqueeze(0).float(), device=DEVICE)

    pbar = tqdm(time_pairs, total=SAMPLING_TIMESTEPS)
    for time, time_next in pbar:
        output = output.detach().requires_grad_(True)
        time_cond = torch.full((1,), time, device=DEVICE, dtype=torch.long)
        pred_noise, x_start, *_ = checkpoint(unet_wrapper, output, time_cond, use_reentrant=False)

        if time_next < 0:
            # Final step: x_start is the clean estimate; skip DDIM transition.
            output = x_start
            continue

        alpha = trainer.ema.ema_model.alphas_cumprod[time]
        alpha_next = trainer.ema.ema_model.alphas_cumprod[time_next]

        # DDIM transition (Song et al. 2020, eq. 12)
        sigma = DDIM_ETA * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma**2).sqrt()

        with torch.no_grad():
            noise = torch.randn_like(output)
            output_p = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # Map from diffusion space [-1, 1] to physical space [0, 1].
        x_start_scaled = ((x_start + 1) * 0.5).clamp(min=EPSILON)

        # Save intermediate estimate every 50 steps.
        if time % 50 == 0:
            np.savez_compressed(
                os.path.join(INTERMEDIATE_DIR, f"x_start_{time}.npz"),
                reconstruction=x_start_scaled[0].detach().cpu().numpy(),
            )

        # DPS guidance: ∇_{x_t} NLL(y | forward_model(x_start))
        # Likelihood: Poisson(λ) + Gaussian(0, η²)  →  NLL ≈ Gaussian with var = λ + η²
        _, lambda_val = forward_imaging(x_start_scaled)
        var = lambda_val + forward_imaging.eta**2 + EPSILON
        res = y - lambda_val
        nll_per_voxel = 0.5 * ((res**2) / var + torch.log(var))
        # mask (out_h, out_w) broadcasts over (num_bins, out_h, out_w)
        nll = (mask * nll_per_voxel).mean()

        grads = torch.autograd.grad(nll, output, create_graph=False)[0]

        lr = LR_MULTIPLIER * (time / SAMPLING_TIMESTEPS)
        output = output_p - lr * grads

        with torch.no_grad():
            x_start_scaled[x_start_scaled < RECON_THRESHOLD] = 0
            norm = (x_start_scaled - ground_truth.unsqueeze(0)).abs().mean()

        pbar.set_postfix(
            norm=norm.item(),
            lr=f"{lr:.0f}",
            nll=f"{nll.item() / (mask.sum() + EPSILON):.4f}",
            t=time,
        )

        del nll, grads, output_p, norm, lambda_val, var, res
        del x_start, x_start_scaled, pred_noise

        if time % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[t={time}] allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    output = ((output.detach() + 1) * 0.5)
    recon = output[0].cpu().numpy()
    recon[recon < RECON_THRESHOLD] = 0

    np.save(os.path.join(RESULT_DIR, "mask.npy"), mask.cpu().numpy())
    np.save(os.path.join(RESULT_DIR, "input_data.npy"), y.cpu().numpy())
    np.save(os.path.join(RESULT_DIR, "reconstruction.npy"), recon)

    contorno = (ground_truth.sum(0) > 0).float().cpu().numpy()
    normalized_input = y / (y.max(0)[0] + EPSILON)
    plot_results(normalized_input * mask, recon * contorno, ground_truth)


if __name__ == "__main__":
    main()
