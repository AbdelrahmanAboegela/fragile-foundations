"""Morphology-preserving perturbations for H&E patches (proposal §10.1).

All functions take a PIL.Image (RGB, uint8) and return a PIL.Image of the
same size.

MAGNITUDE CALIBRATION (2026-09-16, corrected after an audit caught this
docstring overclaiming uniformity — verify against PERTURBATIONS/
STAIN_MAGNITUDE_SWEEP below before trusting any prose description,
including this one):

Five of the seven entries in PERTURBATIONS — brightness, contrast, gamma,
blur, scanner_color — are root-found via scipy.optimize.brentq in
src/calibrate_all_perturbations.py to hit SSIM=0.90 on a 40-patch IDC
calibration set, not hand-picked. That target itself is not arbitrary: it
is the measured mean SSIM of JPEG at quality=70, the 15.5x-compression
point cited as the clinical diagnostic-safety boundary (AJCP 2011).
PERTURBATIONS['jpeg'] is jpeg_q70 itself, so it sits
exactly at that same anchor by construction.

PERTURBATIONS['stain_shift'] (macenko_stain_shift) is CALIBRATED DIFFERENTLY
from the other six, and for a different reason than before (revised
2026-09-16 after a stain-vector estimation bug was found and fixed — see
_macenko_stain_vectors' docstring). With correctly-conditioned stain
vectors, a real Macenko stain shift is nearly SSIM-invisible even at large
nominal concentration scaling (measured on our own 40-patch calibration
set: SSIM=0.985 at s=0.05 up to SSIM=0.85 only at s~=0.80, i.e. h=1.80/
e=0.20 — a physically extreme, unrealistic stain-channel imbalance).
SSIM under-detects color/stain problems because it is dominated by
structural similarity, not color; forcing an SSIM=0.90 (or 0.85) target
here would mean cranking concentration scaling to an unrealistic magnitude
just to hit a number, defeating the point of "morphology-preserving."

So stain_shift's magnitude is NOT SSIM-target-searched like the other six.
It is fixed at two magnitudes in the same order of magnitude as Tellez et
al.'s published stain-augmentation strengths ("Quantifying the effects of
data augmentation and stain color normalization...", Medical Image
Analysis 2019, arXiv:1902.06543): their HED
augmentation draws a per-channel multiplicative factor alpha ~ U(1-sigma,
1+sigma) INDEPENDENTLY for H and E, with sigma=0.05 ("HED-light") and
sigma=0.2 ("HED-strong"). CORRECTION (2026-09-16, second-audit-caught): an
earlier version of this docstring wrongly called Tellez's parametrization
"additive alpha/beta" — it is multiplicative, the same operation as our
h_scale/e_scale. But ours is NOT range-equivalent to theirs: we apply a
FIXED, perfectly anti-correlated shift (h_scale=1+s and e_scale=1-s
together, every time), not Tellez's independent random per-channel draw —
ours is closer to Tellez's worst-case corner than a typical sample from
their distribution, which is arguably appropriate for a robustness stress
test but must not be described as equivalent to their augmentation range.
We use s=0.10 ("mild") and s=0.30 ("moderate") — between and somewhat past
their light/strong sigma values respectively; treat this as "same order of
magnitude as a published precedent," not a literal replication.

IMPORTANT CAVEAT on top of that (2026-09-16, second-audit finding): at low
s, most of stain_shift's measured SSIM loss vs. the clean image is NOT the
stain scaling — it's the 2-stain decompose/reconstruct round-trip itself,
which is lossy even at h_scale=e_scale=1.0 (identity; see
macenko_stain_shift_identity). Measured on our 40-patch calibration set:
identity-scale SSIM vs. clean = 0.9852 (this is the reconstruction noise
floor, not a stain effect). At s=0.10, mean SSIM vs. clean = 0.9831, but
SSIM vs. the IDENTITY-RECONSTRUCTED image (isolating only the actual
scaling step) = 0.9958 — i.e. ~88% of s=0.10's apparent change vs. clean is
reconstruction noise, not stain shift. At s=0.30, SSIM vs. clean = 0.9612,
SSIM vs. identity = 0.9704 — real stain-scaling signal dominates there
(~62%), noise floor contributes less (~38%). CONSEQUENCE: s=0.30
("moderate") is the more scientifically interpretable magnitude for
isolating a genuine stain-shift effect; s=0.10 ("mild") results should be
reported alongside stain_shift_identity's GPFR (see STAIN_MAGNITUDE_SWEEP
and main_experiment.py's --stain-sweep flag) so the reconstruction-noise
contribution can be subtracted out or at least disclosed, not implied to
be a pure stain effect.

Any report using GPFR across the perturbation suite together must say
explicitly that stain_shift's magnitude is not SSIM-matched to the rest —
SSIM is not a valid common yardstick across a stain perturbation and
structural/compression perturbations, and pretending otherwise would
repeat the now-fixed mistake of forcing a shared target that only "worked"
because of a bug.

The original hand-picked constants (factor=1.2 for brightness/contrast/
gamma, radius=1.0 for blur, rgb_gain=(1.05,0.98,1.02) for scanner color)
turned out to be wildly inconsistent in actual visual effect once measured
— e.g. "factor=1.2" was near SSIM=0.90 for brightness but nowhere close for
contrast or gamma, which needed much larger factors to match. Do not revert
to hand-picked constants; if you need a different target SSIM, re-run the
calibration scripts rather than eyeballing new numbers.
"""
from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def brightness_shift(img: Image.Image, factor: float = 1.16747) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast_shift(img: Image.Image, factor: float = 1.73081) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def gamma_correction(img: Image.Image, gamma: float = 2.02228) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


def he_stain_shift(img: Image.Image, h_scale: float = 1.15, e_scale: float = 0.9) -> Image.Image:
    """DEPRECATED crude stand-in — kept only for backward comparison against
    earlier results. Use macenko_stain_shift for anything reported in the
    paper. See PERTURBATIONS dict below for which one is actually wired in.
    """
    arr = np.asarray(img).astype(np.float32)
    od = -np.log((arr + 1) / 256.0)
    od[..., 0] *= h_scale
    od[..., 2] *= e_scale
    out = 256.0 * np.exp(-od) - 1
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _macenko_stain_vectors(od: np.ndarray, alpha_percentile: float = 1.0) -> np.ndarray:
    """Estimate H&E stain vectors via the canonical Macenko procedure (Macenko
    et al., IEEE ISBI 2009, doi:10.1109/ISBI.2009.5193250).

    od: (N, 3) optical density values for N pixels (already background-thresholded).
    Returns a (3, 2) matrix: columns are the estimated hematoxylin and eosin
    stain vectors in OD space.

    TWO-PART FIX (2026-09-16, audit-caught; BOTH parts required — a first
    attempt that only added centering got stuck at 1.03 degrees / cond 111,
    barely better than the original 0.28 degrees / cond 415):

    1. The PLANE must come from the CENTERED OD covariance. Without
       centering, the leading singular vector is the mean-OD direction, not
       a direction of stain variance, and both "stain vectors" collapse
       onto it.
    2. The PROJECTION for finding the extreme angles must use the
       UNCENTERED od. Stain vectors are rays from the OD origin —
       projecting centered data spreads angles over the full circle, so
       the alpha/(100-alpha) percentiles straddle the +/-pi wrap and
       resolve to nearly the same direction. This is the trap an
       incomplete "just add centering" fix falls into (that first attempt).

    Axis order is eigh's (ascending eigenvalues -> [secondary, dominant]),
    NOT svd's vh[:2].T ([dominant, secondary]) — not cosmetic, it sets
    which component is x in arctan2(y, x). The svd order re-triggers angle
    wrap on some patches (measured p5=0.26 degrees, max cond 498 over 60
    patches); the eigh order is stable (measured p5=12.56 degrees, max cond
    11.89 over 200 patches across all 4 IDC samples).

    Verified on real IDC patches: patch 0 of NCBI783.h5 gives 29.08 degrees
    / cond 3.85, H=[0.609,0.650,0.455], E=[0.254,0.936,0.244], against
    literature H=[0.65,0.70,0.29] / E=[0.07,0.99,0.11] (39.4 degrees apart).
    """
    od_mean = od.mean(axis=0)
    _, eigvecs = np.linalg.eigh(np.cov((od - od_mean).T))
    plane = eigvecs[:, 1:3]  # (3, 2): [secondary, dominant] — eigh order, see docstring
    # eigh's eigenvector signs are arbitrary; pin the dominant axis to the
    # positive-OD halfspace so the angle percentiles can't wrap.
    if (od @ plane[:, 1]).mean() < 0:
        plane = plane * np.array([1.0, -1.0])
    proj = od @ plane  # UNCENTERED od — see docstring point 2
    min_angle, max_angle = np.percentile(np.arctan2(proj[:, 1], proj[:, 0]),
                                          [alpha_percentile, 100 - alpha_percentile])
    v1 = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v2 = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])
    # OD is non-negative by construction, so flip any vector pointing away from it.
    if v1.sum() < 0:
        v1 = -v1
    if v2.sum() < 0:
        v2 = -v2
    # Convention: hematoxylin (blue-purple, larger red OD component) first.
    stain_vectors = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)
    norms = np.linalg.norm(stain_vectors, axis=0, keepdims=True)
    return stain_vectors / (norms + 1e-8)


def macenko_stain_shift(img: Image.Image, h_scale: float = 1.10, e_scale: float = 0.90,
                          beta: float = 0.15) -> Image.Image:
    """Real Macenko-based H&E stain perturbation (Macenko et al. 2009)
    — deconvolves the image into hematoxylin/eosin
    concentration channels via properly-estimated stain vectors (see
    _macenko_stain_vectors' two-part fix docstring), scales each channel
    independently, then reconstructs.

    h_scale/e_scale=1.10/0.90 ("mild", s=0.10): a literature-grounded
    magnitude, NOT an SSIM-target search — see the module docstring's
    MAGNITUDE CALIBRATION section for why (with correctly-conditioned
    stain vectors, SSIM=0.85-0.90 is only reachable at physically extreme,
    unrealistic concentration scaling) and for the Tellez et al. precedent
    this magnitude brackets. Measured (not targeted) mean SSIM at this
    magnitude on our 40-patch calibration set: 0.983.

    No percentile clipping of concentrations here (removed 2026-09-16): with
    properly-conditioned stain vectors, identity-scale mean SSIM measured
    0.9852 without clipping vs 0.9822 with it on our 40-patch set — clipping
    now costs more than it returns. (This exact pair is measurement-
    dependent — two independent re-measurements this session got slightly
    different absolute values depending on patch sampling — but the
    DIRECTION, clipping costs SSIM once the real bug is fixed, reproduced
    every time it was checked.) Clipping was only needed to paper over the
    degenerate-basis bug, not an independent fix in its own right.
    """
    arr = np.asarray(img).astype(np.float64)
    od = -np.log((arr.reshape(-1, 3) + 1) / 256.0)
    mask = (od > beta).any(axis=1)
    od_thresh = od[mask] if mask.sum() > 10 else od  # fall back if patch is nearly all background

    try:
        stain_vectors = _macenko_stain_vectors(od_thresh)
        concentrations, _, _, _ = np.linalg.lstsq(stain_vectors, od.T, rcond=None)
    except np.linalg.LinAlgError:
        return he_stain_shift(img, h_scale, e_scale)  # degenerate patch (e.g. flat color) — fall back

    # Only scale FOREGROUND (tissue) pixel concentrations, not background —
    # real inter-lab staining variation changes how STAINED tissue looks,
    # not blank slide background. (This remains correct/motivated
    # independent of the stain-vector fix above.)
    concentrations[0, mask] *= h_scale
    concentrations[1, mask] *= e_scale
    od_reconstructed = (stain_vectors @ concentrations).T
    out = 256.0 * np.exp(-od_reconstructed) - 1
    return Image.fromarray(out.reshape(arr.shape).clip(0, 255).astype(np.uint8))


def macenko_stain_shift_moderate(img: Image.Image) -> Image.Image:
    """Second calibration point for the magnitude sweep, below.

    h_scale/e_scale=1.30/0.70 ("moderate", s=0.30) — see macenko_stain_shift's
    docstring and the module docstring's MAGNITUDE CALIBRATION section:
    this is a literature-grounded magnitude (just past Tellez et al.'s
    HED-strong precedent), not an SSIM-target search. Measured (not
    targeted) mean SSIM on our 40-patch calibration set: 0.961.
    """
    return macenko_stain_shift(img, h_scale=1.30, e_scale=0.70)


def mild_blur(img: Image.Image, radius: float = 1.06062) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg_compression(img: Image.Image, quality: int = 50) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def jpeg_q30(img: Image.Image) -> Image.Image:
    return jpeg_compression(img, quality=30)


def jpeg_q50(img: Image.Image) -> Image.Image:
    return jpeg_compression(img, quality=50)


def jpeg_q70(img: Image.Image) -> Image.Image:
    return jpeg_compression(img, quality=70)


def jpeg_q90(img: Image.Image) -> Image.Image:
    return jpeg_compression(img, quality=90)


def scanner_color_shift(img: Image.Image, strength: float = 5.66476) -> Image.Image:
    """`strength` scales a fixed RGB-gain direction (0.05, -0.02, 0.02) —
    the original hand-picked (1.05, 0.98, 1.02) gain, now reparametrized as
    direction * strength so it has a single calibratable magnitude knob.
    Default is root-found (src/calibrate_all_perturbations.py) to hit
    SSIM=0.90, not guessed — see the module docstring above. This module-
    level default is the canonical IDC (core-dataset) calibration, pooled
    across all 4 of IDC's samples (2026-09-16 multi-sample fix) — kept in
    sync with configs/perturbation_calibration.json['IDC'] by hand; if you
    re-run calibration, update this default too so example figures
    (generate_example_figures.py doesn't read the JSON) stay consistent
    with what the actual experiments use.
    """
    base_direction = np.array([0.05, -0.02, 0.02])
    gains = 1.0 + strength * base_direction
    arr = np.asarray(img).astype(np.float32)
    for c in range(3):
        arr[..., c] *= gains[c]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


PERTURBATIONS = {
    "brightness": brightness_shift,
    "contrast": contrast_shift,
    "gamma": gamma_correction,
    # FIX (2026-09-17, adversarial-review-caught): this pointed at
    # macenko_stain_shift (the "mild" s=0.10 default) while every paper
    # draft claimed stain_shift results were reported at "moderate" —
    # the code never matched that claim. Moderate (s=0.30) is also the
    # more scientifically defensible choice per the noise-floor analysis
    # in macenko_stain_shift's own docstring (~62% real signal at
    # moderate vs. ~88% reconstruction-noise at mild) — so fix the CODE
    # to match the INTENT, not the other way around.
    "stain_shift": macenko_stain_shift_moderate,  # was he_stain_shift (crude RGB-OD approx) — fixed 2026-09-16; was mild-not-moderate — fixed 2026-09-17
    "blur": mild_blur,
    "jpeg": jpeg_q70,  # was jpeg_compression (quality=50, SSIM~0.876) — switched to q70 (SSIM=0.8999968,
                       # the exact anchor value) so this entry is genuinely part of the SSIM=0.90-matched
                       # suite instead of silently sitting outside it. Audit-flagged 2026-09-16.
    "scanner_color": scanner_color_shift,
}

# Magnitude-calibration check (research-flagged concern, 2026-09-16): does
# the earlier stain_shift null (GPFR=0.000) reflect genuine robustness, or a
# too-weak perturbation magnitude relative to real inter-lab variation
# (PLISM protocol)? Run this as a separate, explicit
# comparison against PERTURBATIONS['stain_shift'], not folded into it, so
# both magnitudes are reported rather than silently swapped.
def macenko_stain_shift_identity(img: Image.Image) -> Image.Image:
    """Null control (added 2026-09-16, audit-recommended): runs the full
    Macenko decompose-reconstruct pipeline with h_scale=e_scale=1.0, i.e. no
    actual stain scaling. Any nonzero GPFR here would mean the
    decompose/reconstruct round-trip itself is destructive/lossy
    independent of the scaling step — this entry isolates that possibility
    so it isn't conflated with a genuine stain-shift effect.
    """
    return macenko_stain_shift(img, h_scale=1.0, e_scale=1.0)


STAIN_MAGNITUDE_SWEEP = {
    "stain_shift_identity": macenko_stain_shift_identity,  # h=1.0/e=1.0, null control — isolates reconstruction-roundtrip noise from real stain-shift effect
    "stain_shift_mild": macenko_stain_shift,               # h=1.10/e=0.90, measured SSIM~0.983 (literature-grounded, not SSIM-targeted)
    "stain_shift_moderate": macenko_stain_shift_moderate,  # h=1.30/e=0.70, measured SSIM~0.961 (literature-grounded, not SSIM-targeted)
}

# Quality-level sweep for the JPEG dose-response analysis. NOTE (updated
# 2026-09-16): JPEG was briefly the apparent "dominant failure mode" under
# an earlier, leakage-affected analysis — that ranking did not survive a
# proper held-out/cross-validated re-run (see main_experiment_results.csv),
# and a separate stain_shift result that briefly replaced it as "dominant"
# was itself traced to a degenerate Macenko stain-vector basis (see
# _macenko_stain_vectors' fix history above), not a real effect. Treat any
# cross-perturbation ranking as provisional until re-verified post-fix — the
# scientific content of THIS sweep (does GPFR become non-trivial within the
# compression range clinicians call diagnostically safe: ~15-20x, PIL
# quality 70-90) stands on its own regardless of how JPEG ranks against
# other perturbations.
JPEG_SWEEP = {
    "jpeg_q30": jpeg_q30,
    "jpeg_q50": jpeg_q50,
    "jpeg_q70": jpeg_q70,
    "jpeg_q90": jpeg_q90,
}
