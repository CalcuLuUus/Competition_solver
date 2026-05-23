import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss, ssim as slow_ssim

try:
    from fused_ssim import fused_ssim as fast_ssim
except ImportError:
    fast_ssim = None


def sampling_cameras(viewpoint_stack: Sequence, num_cams: int = 10) -> List:
    """Randomly pick a subset of cameras from the current training queue."""
    num_cams = min(num_cams, len(viewpoint_stack))
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(viewpoint_stack) - 1)
        camlist.append(viewpoint_stack.pop(loc))
    return camlist


def get_loss(reconstructed_image: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
    """Compute the normalized per-pixel L1 error map used to flag hard regions."""
    l1 = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    denom = torch.clamp(torch.max(l1) - torch.min(l1), min=1e-6)
    l1_loss_norm = (l1 - torch.min(l1)) / denom
    return l1_loss_norm


def compute_photometric_loss(reconstructed_image: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
    """Blend L1 and SSIM losses to estimate the per-view reconstruction quality."""
    Ll1 = l1_loss(reconstructed_image, original_image)
    if fast_ssim is not None:
        ssim_value = fast_ssim(reconstructed_image.unsqueeze(0), original_image.unsqueeze(0))
    else:
        ssim_value = slow_ssim(reconstructed_image, original_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)
    return loss


def compute_gaussian_score_fastgs(
    camlist: Sequence,
    gaussians,
    pipe,
    bg: torch.Tensor,
    args,
    render_scale: float = 1.0,
    use_trained_exp: bool = False,
    densify: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute multi-view consistency scores for VCD/VCP."""
    if len(camlist) == 0:
        return None, None

    full_metric_counts = None
    full_metric_score = None

    for cam in camlist:
        gt_image = cam.original_image.cuda()
        image_height = int(cam.image_height)
        image_width = int(cam.image_width)
        if render_scale > 1:
            target_size = (
                max(1, int(image_height / render_scale)),
                max(1, int(image_width / render_scale)),
            )
            gt_image = F.interpolate(
                gt_image[None],
                size=target_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        else:
            target_size = (image_height, image_width)

        # Render once to compute view-specific losses and error map.
        render_pkg = render_fastgs(
            cam,
            gaussians,
            pipe,
            bg,
            args.fastgs_mult,
            use_trained_exp=use_trained_exp,
            render_size=target_size,
        )
        render_image = render_pkg["render"]
        photometric_loss = compute_photometric_loss(render_image, gt_image)

        l1_loss_norm = get_loss(render_image, gt_image)
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        # Re-render to retrieve per-Gaussian hit counts within the error mask.
        render_pkg = render_fastgs(
            cam,
            gaussians,
            pipe,
            bg,
            args.fastgs_mult,
            get_flag=True,
            metric_map=metric_map,
            use_trained_exp=use_trained_exp,
            render_size=target_size,
        )
        accum_loss_counts = render_pkg["accum_metric_counts"]

        if densify:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        per_gaussian_score = photometric_loss * accum_loss_counts
        if full_metric_score is None:
            full_metric_score = per_gaussian_score.clone()
        else:
            full_metric_score += per_gaussian_score

    if full_metric_score is None:
        return None, None

    score_min = torch.min(full_metric_score)
    denom = torch.max(full_metric_score) - score_min + 1e-6
    pruning_score = (full_metric_score - score_min) / denom

    if densify and full_metric_counts is not None:
        importance_score = torch.div(
            full_metric_counts, len(camlist), rounding_mode="floor"
        )
    else:
        importance_score = None

    return importance_score, pruning_score
