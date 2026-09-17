"""Image-preservation and robustness metrics (proposal §10.4, §7.5, §13.2).

SSIM/LPIPS gate whether a perturbation counts as "morphology-preserving".
GPFR is the core proposed contribution metric.
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch_msssim import ssim as _ssim
import lpips as _lpips_lib

_lpips_model = None


def _get_lpips_model(net: str = "alex", device: str = "cpu"):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = _lpips_lib.LPIPS(net=net).to(device)
        _lpips_model.eval()
    return _lpips_model


def compute_ssim(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """img_a, img_b: (1, 3, H, W) float tensors in [0, 1]."""
    return _ssim(img_a, img_b, data_range=1.0).item()


_lpips_unavailable = False  # set True after the first failure, so we stop retrying a dead network path


def compute_lpips(img_a: torch.Tensor, img_b: torch.Tensor, device: str = "cpu") -> float | None:
    """img_a, img_b: (1, 3, H, W) float tensors in [-1, 1] (LPIPS convention).

    Returns None (not a crash) if LPIPS's pretrained backbone can't be
    fetched — it needs a one-time 233MB AlexNet download from
    download.pytorch.org, and that host stalled/failed outright on this
    network on 2026-09-15 (confirmed independently via a direct `curl`
    timeout, not just this library). SSIM alone (no network dependency,
    already computed) is sufficient to gate morphology-preservation for the
    kill test; LPIPS is a nice-to-have supplementary check for the
    main-paper experiments once run somewhere with reliable access to that
    host, not a hard requirement for the go/no-go decision here.
    """
    global _lpips_unavailable
    if _lpips_unavailable:
        return None
    try:
        model = _get_lpips_model(device=device)
    except Exception as e:
        print(f"[warn] LPIPS backbone unavailable ({e}) — falling back to SSIM-only "
              "for the rest of this run.")
        _lpips_unavailable = True
        return None
    with torch.no_grad():
        d = model(img_a.to(device), img_b.to(device))
    return d.item()


def color_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Mean per-pixel Euclidean RGB distance, 0-255 scale inputs."""
    return float(np.sqrt(((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2).sum(-1)).mean())


def perturbation_norm(img_a: np.ndarray, img_b: np.ndarray, p: int = 2) -> float:
    diff = (img_a.astype(np.float32) - img_b.astype(np.float32)).ravel()
    return float(np.linalg.norm(diff, ord=p))


def gene_program_flip_rate(
    clean_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    method: str = "per_sample_relative",
    z_thresh: float = 1.0,
) -> float | None:
    """GPFR per proposal §7.5, with the per-cohort/per-sample threshold fix
    from the review (do NOT use one global absolute cutoff across organs).

    NOVELTY STATUS (2026-09-16, added after being asked directly whether
    GPFR is a novel metric — the honest answer, not a marketing one):
    GPFR is NOT a novel metric CLASS. "Does a model's output flip under a
    perturbation of the same input" is an established instability-metric
    family in clinical ML.'s "Decision-flip-rate metric
    family" entry (arXiv 2603.00192, arXiv 2606.07929). What's new here is
    the SPECIFIC INSTANTIATION: applying flip-rate instability to Hallmark-
    aggregated GENE-PROGRAM scores predicted from histology (not to a
    classifier's top-1 label or an LLM's answer, which is what the cited
    precedents measure), which is a genuinely different measurement target
    with different statistical properties (a continuous cohort-relative
    score, not a discrete class label). Cite this as "an instantiation of a
    precedented metric family, adapted to a new measurement target," not as
    an invented-from-scratch metric — the honest framing survives review
    better than an oversold one.

    THE EQUATION: for one (organ, perturbation, program) triple, with N
    held-out samples, cohort_std = std(clean_scores) across all N samples:
        z_i = |perturbed_score_i - clean_score_i| / (cohort_std + 1e-8)
        flipped_i = 1 if z_i > z_thresh else 0
        GPFR = mean(flipped_i) over samples with a valid (non-NaN) z_i

    WHY PER-SAMPLE-RELATIVE, NOT A GLOBAL ABSOLUTE CUTOFF: raw program-score
    magnitudes are not comparable across organs/programs/gene-panel sizes
    (a 2-gene vs.
    200-gene program score has a totally different natural scale). Defining
    "flipped" relative to THIS cohort's own natural score spread avoids
    needing a separately-tuned absolute threshold per organ/program, at the
    cost of GPFR values not being directly comparable in absolute terms
    across cohorts with very different score distributions either — a
    genuine, acknowledged tradeoff, not a free lunch.

    WHY z_thresh=1.0: this is a conventional, interpretable choice ("more
    than one cohort standard deviation of movement") but — stated plainly,
    not glossed over — it was NOT separately derived from a formal power
    analysis or optimized against any external criterion. Report a
    sensitivity sweep across a few z_thresh values (e.g. 0.5/1.0/1.5/2.0) as
    a robustness check in any write-up that leans on a specific GPFR number
    mattering (src/sensitivity_z_thresh.py) — don't present z_thresh=1.0 as
    more validated than it actually is.

    clean_scores, perturbed_scores: 1D arrays, same length, one program's
    score per sample, ALREADY restricted to one cohort/organ if using
    per-cohort z-scoring (don't mix organs into one z-score computation).

    method="per_sample_relative": a sample is "flipped" if the perturbed
    score crosses more than `z_thresh` cohort-standard-deviations away from
    its own clean score.
    """
    assert clean_scores.shape == perturbed_scores.shape
    if np.all(np.isnan(clean_scores)):
        # A program with zero gene overlap for this organ/panel gets scored
        # as all-NaN upstream (gene_program_scoring.score_programs) — NaN
        # comparisons silently evaluate False in numpy, which would otherwise
        # produce a misleading GPFR of 0.0 ("perfectly robust") when really
        # there was nothing to measure. Surface that as None/NaN instead so
        # it can't be mistaken for a real robustness result. Found 2026-09-15
        # while generalizing the kill test across organs with different gene
        # panels — some organs won't cover all 5 target programs.
        return None
    if method == "per_sample_relative":
        # PARTIAL-NaN fix (audit-flagged, 2026-09-16): the all-NaN check
        # above only caught the case where EVERY sample is NaN. If only
        # SOME samples are NaN (e.g. a numerical edge case in a specific
        # patch's prediction), plain .std() returns NaN, making every z
        # comparison silently False -> GPFR=0.0 -- the exact misleading
        # "perfectly robust" artifact this function exists to prevent, just
        # via a different path. Use nanstd/nanmean and drop NaN samples from
        # the denominator instead of letting them poison the whole result.
        valid = ~(np.isnan(clean_scores) | np.isnan(perturbed_scores))
        if valid.sum() < len(clean_scores):
            n_dropped = len(clean_scores) - valid.sum()
            print(f"[warn] gene_program_flip_rate: dropping {n_dropped}/{len(clean_scores)} "
                  f"samples with NaN score (partial-NaN case)")
        if valid.sum() == 0:
            return None
        cohort_std = np.nanstd(clean_scores[valid]) + 1e-8
        z = np.abs(perturbed_scores[valid] - clean_scores[valid]) / cohort_std
        flipped = z > z_thresh
    else:
        raise NotImplementedError(f"Unknown GPFR method: {method}")
    return float(flipped.mean())
