# SPDX-License-Identifier: OpenMDW-1.1
"""Whole-sequence prediction videos using EveryNDrawSample's MP4 writer."""

from pathlib import Path

import torch

from cosmos_framework.tools.visualize.video import save_img_or_video


@torch.no_grad()
def save_latent_video(model, latent, path):
    """Decode L0 plus all selected predictions once, then save at the callback's 16 FPS.

    Only the initial observed latent belongs in this sequence. Later observed
    latents remain inference conditions and never replace the saved predictions.
    """
    path = Path(path)
    if path.suffix != ".mp4":
        raise ValueError("Prediction video path must end in .mp4")
    decoded = model.decode(latent.to(device=model.tensor_kwargs_fp32["device"])).float().cpu()
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[2] < 2:
        raise ValueError("Prediction video must decode to [1,C,T,H,W] with T > 1")
    if not torch.isfinite(decoded).all():
        raise RuntimeError("VAE returned nonfinite predicted video")
    # Match EveryNDrawSample.run_save, including its float-to-uint8 conversion
    # and ImageIO quality defaults rather than specifying a separate CRF/preset.
    pixels = (1.0 + decoded[0].clamp(-1, 1)) / 2.0
    path.parent.mkdir(parents=True, exist_ok=True)
    save_img_or_video(pixels, str(path.with_suffix("")), fps=16)
    return dict(video_path=str(path), video_frames=decoded.shape[2], video_fps=16)
