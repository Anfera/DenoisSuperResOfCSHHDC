import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips

from canopyPlots import createCHM


def _to_numpy(arr):
    """Convert a tensor-like object to a NumPy array without gradients."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def plot_results(input_image, sample, gt_image, show: bool = True) -> dict[str, float]:
    """
    Plot CHM/DTM/profiles and return reconstruction metrics.

    Args:
        input_image: Input tensor or array.
        sample: Reconstructed tensor or array.
        gt_image: Ground-truth tensor or array.
        show: Whether to display the matplotlib figure.

    Returns:
        Dictionary of computed metrics.
    """
    input_np = _to_numpy(input_image)
    sample_np = _to_numpy(sample)
    gt_np = _to_numpy(gt_image)

    chm_input, dtm_input, hillshade_input, _ = createCHM(input_np, percentile=0.98)
    chm_recon, dtm_recon, hillshade_recon, _ = createCHM(sample_np, percentile=0.98)
    chm_gt, dtm_gt, hillshade_gt, _ = createCHM(gt_np, percentile=0.98)

    chm_input = chm_input * 0.5
    chm_recon = chm_recon * 0.5
    chm_gt = chm_gt * 0.5

    dtm_input = dtm_input * 0.5
    dtm_recon = dtm_recon * 0.5
    dtm_gt = dtm_gt * 0.5

    fig, axs = plt.subplots(3, 4, figsize=(15, 15))

    axs[0, 0].imshow(chm_input, cmap="viridis")
    axs[0, 0].set_aspect(0.5)
    axs[0, 0].set_title("CHM Input")

    axs[0, 1].imshow(chm_recon, cmap="viridis", vmin=chm_gt.min(), vmax=chm_gt.max())
    axs[0, 1].set_title("CHM Reconstruction")

    axs[0, 2].imshow(chm_gt, cmap="viridis")
    axs[0, 2].set_title("CHM Ground Truth")

    axs[1, 0].imshow(dtm_input, cmap="copper")
    axs[1, 0].imshow(hillshade_input, cmap="Grays", alpha=0.35)
    axs[1, 0].set_aspect(0.5)
    axs[1, 0].set_title("DTM Input")

    axs[1, 1].imshow(dtm_recon, cmap="copper", vmin=dtm_gt.min(), vmax=dtm_gt.max())
    axs[1, 1].imshow(hillshade_recon, cmap="Grays", alpha=0.35)
    axs[1, 1].set_title("DTM Reconstruction")

    axs[1, 2].imshow(dtm_gt, cmap="copper")
    axs[1, 2].imshow(hillshade_gt, cmap="Grays", alpha=0.35)
    axs[1, 2].set_title("DTM Ground Truth")

    profile_index_input = 3 * input_np.shape[1] // 4
    profile_index_gt = 3 * gt_np.shape[1] // 4

    axs[2, 0].imshow(input_np[::-1, profile_index_input, :], cmap="gray_r", interpolation="nearest")
    axs[2, 0].set_aspect(1 / 3)
    axs[2, 0].set_title("Profile Input")

    axs[2, 1].imshow(sample_np[::-1, profile_index_gt, :], cmap="gray_r", interpolation="nearest")
    axs[2, 1].set_title("Profile Reconstruction")

    axs[2, 2].imshow(gt_np[::-1, profile_index_gt, :], cmap="gray_r", interpolation="nearest")
    axs[2, 2].set_title("Profile Ground Truth")

    error_chm = np.abs(chm_recon - chm_gt)
    error_dtm = np.abs(dtm_recon - dtm_gt)
    error_profile = np.abs(sample_np[::-1, profile_index_gt, :] - gt_np[::-1, profile_index_gt, :])

    im0 = axs[0, 3].imshow(error_chm, cmap="turbo")
    axs[0, 3].set_title("CHM Error")
    fig.colorbar(im0, ax=axs[0, 3])

    im1 = axs[1, 3].imshow(error_dtm, cmap="turbo")
    axs[1, 3].set_title("DTM Error")
    fig.colorbar(im1, ax=axs[1, 3])

    im2 = axs[2, 3].imshow(error_profile, cmap="turbo")
    axs[2, 3].set_title("Profile Error")
    fig.colorbar(im2, ax=axs[2, 3])

    metrics = {
        "ssim_chm": ssim(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min(), win_size=11),
        "psnr_chm": psnr(chm_gt, chm_recon, data_range=chm_gt.max() - chm_gt.min()),
        "ssim_dtm": ssim(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min(), win_size=11),
        "psnr_dtm": psnr(dtm_gt, dtm_recon, data_range=dtm_gt.max() - dtm_gt.min()),
    }

    lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="vgg").to(lpips_device)
    metrics["lpips_chm"] = lpips_fn(
        torch.tensor(chm_gt).float().unsqueeze(0).unsqueeze(0).to(lpips_device),
        torch.tensor(chm_recon).float().unsqueeze(0).unsqueeze(0).to(lpips_device),
    ).item()
    metrics["lpips_dtm"] = lpips_fn(
        torch.tensor(dtm_gt).float().unsqueeze(0).unsqueeze(0).to(lpips_device),
        torch.tensor(dtm_recon).float().unsqueeze(0).unsqueeze(0).to(lpips_device),
    ).item()

    metrics["mse"] = torch.mean((torch.tensor(gt_np) - torch.tensor(sample_np)) ** 2).item()
    metrics["mae"] = torch.mean(torch.abs(torch.tensor(gt_np) - torch.tensor(sample_np))).item()

    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return metrics
