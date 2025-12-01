## Diffusion-Based Joint Recovery, Denoising, and Super-Resolution of Compressed-Sensing Satellite LiDAR Data

![Low- vs high-resolution reconstructions](images/LowVSHigh.png)

Final code for the paper pipeline that reconstructs canopy volumes using a diffusion model with a Poisson forward imaging model. The entrypoint is `SingleLikelihood.py`.

### Paper details

- Title: "Diffusion-Based Joint Recovery, Denoising, and Super-Resolution of Compressed-Sensing Satellite LiDAR Data"
- Authors: Andres Ramirez-Jaime (Graduate Student Member, IEEE), Nestor Porras-Diaz (Graduate Student Member, IEEE), Mark Stephen, Guangning Yang, and Gonzalo R. Arce (Life Fellow, IEEE)
- Affiliations: University of Delaware, Dept. of Electrical and Computer Engineering (aramjai@udel.edu; nestorfe@udel.edu; arce@udel.edu) and NASA Goddard Space Flight Center (mark.a.stephen@nasa.gov; guangning.yang-1@nasa.gov)
- Funding: U.S. National Science Foundation Grant No. 2404740 and NASA Grant No. 80NSSC25K7395
- Links: dataset https://huggingface.co/datasets/anfera236/HHDC

### Quickstart

1. Python 3.10+ recommended. Install deps: `pip install -r requirements.txt`.
2. Place inputs in `TestCube/` (`input{RESOLUTION}.npy` and `gt{RESOLUTION}.npy` or the provided `gt2.npy`) and keep `Mblue.tiff` in the project root for the blue-noise mask.
3. Add the pretrained checkpoint at `results/model{RESOLUTION}.pt` (resolution defaults to 2; adjust in `config.py`).
4. Run inference: `python SingleLikelihood.py`. Outputs land in `resultCubes/` (final artifacts) and `intermediateCubesTest/` (DDIM snapshots).

### Configuration

All tunable knobs live in `config.py` (resolution, sampling steps, mask type and ratio, SNR, thresholds, and output locations). Update `CHECKPOINT_TEMPLATE`/`CHECKPOINT_DIR` if your weights live elsewhere.

### Repository Layout

- `SingleLikelihood.py` — main inference script wired to the diffusion model and Poisson forward operator.
- `config.py` — runtime constants, RNG seeding, and checkpoint path helper.
- `data_utils.py` — loading and normalization for the provided cubes.
- `mask_utils.py` — blue-noise / Bayer / random mask creation.
- `visualization.py` — CHM/DTM/profile plots plus SSIM/PSNR/LPIPS/MSE/MAE metrics.
- `forwardImagingPoisson.py` — forward model for Poisson-distributed acquisitions.
- `denoising_diffusion_pytorch/` — trimmed diffusion implementation (only the pieces needed for this pipeline).

### Notes

- `Trainer` now supports `inference_only=True`, so inference does not require the training dataset.
- Generated artifacts (`resultCubes/`, `intermediateCubesTest/`, `results/`) are git-ignored by default.
