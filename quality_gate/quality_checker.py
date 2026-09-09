"""
NETRADRISHTI Quality Gate Module
Performs automated image usability validation for rural fundus cameras:
1. Focus / Blur Detection (Laplacian variance)
2. Illumination Assessment (Underexposure, Overexposure, Glare)
3. Field of View (FOV) & Usable Area Analysis
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from config import (
    BLUR_THRESHOLD_MIN,
    ILLUMINATION_MIN,
    ILLUMINATION_MAX,
    MIN_RETINAL_COVERAGE_PERCENTAGE
)


@dataclass
class QualityResult:
    status: str                         # "IMAGE ACCEPTED" or "UNGRADABLE IMAGE"
    usable: bool                        # True if gradable, False if ungradable
    quality_score: float                # 0 - 100 overall composite score
    blur_score: float                   # Raw Laplacian variance
    illumination_score: float           # Mean luminance (0 - 255)
    fov_score: float                    # Retinal aperture coverage (%)
    reason: Optional[str]               # Specific rejection reason if ungradable
    recommendation: str                 # Guidance for the rural health worker
    focus_status: str = "PASS"          # "PASS" or "FAIL"
    illumination_status: str = "PASS"   # "PASS" or "FAIL"
    fov_status: str = "PASS"            # "PASS" or "FAIL"
    artifacts_status: str = "PASS"      # "PASS" or "FAIL"


def _compute_laplacian_variance(gray: np.ndarray) -> float:
    """Computes Laplacian variance to measure edge sharpness / blur."""
    kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)
    lap = convolve(gray.astype(np.float32), kernel, mode='reflect')
    return float(np.var(lap))


def assess_image_quality(image_input: Union[str, Image.Image, np.ndarray]) -> QualityResult:
    """
    Evaluates retinal fundus image gradability against clinical quality criteria.
    Returns QualityResult with status, sub-scores, and actionable instructions.
    """
    # Load and normalize to RGB Image
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    total_pixels = h * w

    # Standard clinical practice uses the green channel or luminance for vessel contrast
    # Green channel has highest vessel-background contrast in fundus photography
    green_channel = arr[:, :, 1]
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # 1. Field of View / Mask Extraction
    # Retinal fundus images are circular fields surrounded by dark camera borders
    # Identify non-background pixels (intensity > 15)
    retina_mask = luminance > 18.0
    retina_pixel_count = np.sum(retina_mask)
    fov_percentage = float((retina_pixel_count / total_pixels) * 100.0)

    # 2. Illumination Assessment
    if retina_pixel_count > 0:
        mean_retina_lum = float(np.mean(luminance[retina_mask]))
        overexposed_ratio = float(np.sum(luminance[retina_mask] > 240.0) / retina_pixel_count)
    else:
        mean_retina_lum = float(np.mean(luminance))
        overexposed_ratio = 1.0

    # 3. Focus / Blur Assessment
    blur_score = _compute_laplacian_variance(green_channel)

    # Sub-checks status evaluation
    focus_status = "FAIL" if blur_score < BLUR_THRESHOLD_MIN else "PASS"
    illum_status = "FAIL" if (mean_retina_lum < ILLUMINATION_MIN or mean_retina_lum > ILLUMINATION_MAX or overexposed_ratio > 0.28) else "PASS"
    fov_status = "FAIL" if fov_percentage < MIN_RETINAL_COVERAGE_PERCENTAGE else "PASS"
    artifacts_status = "FAIL" if (overexposed_ratio > 0.35 or mean_retina_lum < 25.0) else "PASS"

    # Rejection Logic
    reasons = []
    
    # Check 1: Blur
    if focus_status == "FAIL":
        reasons.append("Image too blurry / Defocused")

    # Check 2: Illumination
    if illum_status == "FAIL":
        if mean_retina_lum < ILLUMINATION_MIN:
            reasons.append("Low illumination (Underexposed)")
        else:
            reasons.append("Poor illumination (Severe flash glare / Overexposed)")

    # Check 3: Field of View
    if fov_status == "FAIL":
        reasons.append("Insufficient field of view (Severe occlusion or clipping)")

    # Overall Gradability Decision
    if len(reasons) > 0:
        primary_reason = reasons[0]
        # Composite score penalization for ungradable images
        comp_score = max(5.0, min(48.0, (blur_score / BLUR_THRESHOLD_MIN * 20.0) + (fov_percentage * 0.25)))
        return QualityResult(
            status="UNGRADABLE IMAGE",
            usable=False,
            quality_score=round(comp_score, 1),
            blur_score=round(blur_score, 2),
            illumination_score=round(mean_retina_lum, 1),
            fov_score=round(fov_percentage, 1),
            reason=primary_reason,
            recommendation="Please recapture the fundus image.",
            focus_status=focus_status,
            illumination_status=illum_status,
            fov_status=fov_status,
            artifacts_status=artifacts_status
        )

    # Gradable / Accepted
    # Normalizing composite score to 50-100 scale
    blur_factor = min(1.0, blur_score / 350.0) * 40.0
    illum_factor = (1.0 - abs(mean_retina_lum - 120.0) / 100.0) * 35.0
    fov_factor = min(1.0, fov_percentage / 75.0) * 25.0
    composite = min(99.0, max(65.0, blur_factor + illum_factor + fov_factor))

    return QualityResult(
        status="IMAGE ACCEPTED",
        usable=True,
        quality_score=round(composite, 1),
        blur_score=round(blur_score, 2),
        illumination_score=round(mean_retina_lum, 1),
        fov_score=round(fov_percentage, 1),
        reason=None,
        recommendation="Image quality verified. Minimum clinical requirements satisfied.",
        focus_status="PASS",
        illumination_status="PASS",
        fov_status="PASS",
        artifacts_status="PASS"
    )
