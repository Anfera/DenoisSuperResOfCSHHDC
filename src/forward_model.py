import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarForwardImagingModel(nn.Module):
    def __init__(
        self,
        input_res_m=(2.0, 2.0),
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,
        eta=0.5,
        ref_altitude=500.0,
        ref_photon_count=20.0,
    ):
        """
        Physics-based LiDAR forward imaging model.

        Simulates the complete acquisition chain: spatial blurring by the
        laser footprint, downsampling to the output resolution, Poisson
        photon-counting noise, and Gaussian readout noise.

        Args:
            input_res_m (tuple): Physical size of input pixels (dy, dx) in metres.
            output_res_m (tuple): Physical size of output pixels (dy, dx) in metres.
            footprint_diameter_m (float): 1/e² beam diameter in metres.
            b (float): Background photon rate per pixel.
            eta (float): Gaussian readout noise standard deviation.
            ref_altitude (float): Reference flight altitude in metres.
            ref_photon_count (float): Target mean photon count at reference altitude.
        """
        super().__init__()
        self.b = b
        self.eta = eta
        self.ref_altitude = ref_altitude
        self.ref_photon_count = ref_photon_count

        self.input_res_m = input_res_m
        self.output_res_m = output_res_m

        # Area scale factor: ratio of output pixel area to input pixel area.
        in_area = input_res_m[0] * input_res_m[1]
        out_area = output_res_m[0] * output_res_m[1]
        self.area_scale_factor = out_area / in_area

        # Gaussian kernel sigma in input pixels.
        # Convention: 1/e² diameter = 4 * sigma.
        sigma_m = footprint_diameter_m / 4.0
        avg_input_res = (input_res_m[0] + input_res_m[1]) / 2.0
        sigma_px = sigma_m / avg_input_res

        # Kernel size: 6*sigma captures >99 % of energy.
        kernel_size = int(math.ceil(6 * sigma_px))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.register_buffer("kernel", self._create_gaussian_kernel(kernel_size, sigma_px))

        print(f"LidarForwardImagingModel: {input_res_m} m/px → {output_res_m} m/px | "
              f"footprint {footprint_diameter_m} m (σ={sigma_m:.2f} m / {sigma_px:.2f} px)")

    def _create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - (size - 1) / 2
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, X_h, altitude=500.0):
        """
        Args:
            X_h: High-resolution input volume, shape (B, num_bins, H, W) or (num_bins, H, W).
            altitude (float): Current flight altitude in metres.

        Returns:
            Y_l (Tensor): Noisy observations (Poisson + Gaussian), same batch shape as X_h.
            lambda_val (Tensor): Poisson rate parameter λ (deterministic), same shape as Y_l.
        """
        squeeze = X_h.ndim == 3
        if squeeze:
            X_h = X_h.unsqueeze(0)

        batch_size, num_bins, h_in, w_in = X_h.shape

        # --- 1. Dynamic output size from field-of-view ---
        fov_h_m = h_in * self.input_res_m[0]
        fov_w_m = w_in * self.input_res_m[1]
        out_h = int(fov_h_m / self.output_res_m[0])
        out_w = int(fov_w_m / self.output_res_m[1])

        # --- 2. Physics normalisation ---
        energy_per_tube = X_h.sum(dim=1, keepdim=True)
        global_mean_energy = energy_per_tube.mean(dim=(2, 3), keepdim=True)
        X_norm = X_h / (global_mean_energy + 1e-8)

        dist_scale = (self.ref_altitude / altitude) ** 2
        target_intensity = (self.ref_photon_count / self.area_scale_factor) * dist_scale
        X_scaled = X_norm * target_intensity

        # --- 3. Spatial blurring (depthwise convolution) ---
        current_kernel = self.kernel.repeat(num_bins, 1, 1, 1)
        padding = current_kernel.shape[-1] // 2
        X_blurred = F.conv2d(X_scaled, current_kernel, padding=padding, groups=num_bins)

        # --- 4. Downsampling to output resolution ---
        X_binned = F.interpolate(X_blurred, size=(out_h, out_w), mode="area")
        X_integrated = X_binned * self.area_scale_factor

        # --- 5. Noise ---
        lambda_val = torch.relu(X_integrated) + self.b
        X_l = torch.poisson(lambda_val)
        Y_l = X_l + torch.randn_like(X_l) * self.eta

        if squeeze:
            Y_l = Y_l.squeeze(0)
            lambda_val = lambda_val.squeeze(0)

        return Y_l, lambda_val


def estimate_b_eta(Y, patch_size=8, step=None, retain_fraction=0.1):
    """
    Robustly estimate background rate b and readout noise eta from
    observed waveform data Y ~ Poisson(S + b) + N(0, eta²).

    Filters structural variance by fitting only the "flattest" patches
    (those whose variance is closest to the Poisson expectation).

    Args:
        Y: Observation array or tensor, shape (..., H, W).
        patch_size (int): Spatial patch size for local statistics.
        step (int | None): Stride between patches (defaults to patch_size // 2).
        retain_fraction (float): Fraction of flattest patches used for eta fitting.

    Returns:
        b_est (float): Estimated background photon rate.
        eta_est (float): Estimated readout noise standard deviation.
    """
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    Y = np.atleast_3d(Y)
    if Y.ndim == 4:
        Y = Y.reshape(-1, Y.shape[-2], Y.shape[-1])

    N_imgs, h, w = Y.shape
    step = step or max(1, patch_size // 2)

    means, vars_list = [], []
    for n in range(N_imgs):
        img = Y[n]
        for i in range(0, h - patch_size + 1, step):
            for j in range(0, w - patch_size + 1, step):
                patch = img[i : i + patch_size, j : j + patch_size]
                means.append(float(np.mean(patch)))
                vars_list.append(float(np.var(patch, ddof=1)))

    means = np.array(means)
    vars_ = np.array(vars_list)

    if len(means) < 20:
        print("Warning: Not enough patches for robust estimation.")
        return 0.0, 0.0

    b_est = max(0.0, np.percentile(means, 3.0))

    diff = vars_ - means
    num_to_keep = max(10, int(len(means) * retain_fraction))
    flat_indices = np.argsort(diff)[:num_to_keep]
    flat_means = means[flat_indices]
    flat_vars = vars_[flat_indices]

    A = np.vstack([flat_means, np.ones_like(flat_means)]).T
    slope, intercept = np.linalg.lstsq(A, flat_vars, rcond=None)[0]
    eta_est = np.sqrt(max(0.0, intercept))

    print(f"Robust estimate → b={b_est:.4f}, eta={eta_est:.4f} "
          f"(fit slope={slope:.3f} on {num_to_keep} patches)")
    return b_est, eta_est
