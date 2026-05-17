"""
DCW – Differential Correction in Wavelet domain
CWM – CFG Wavelet Mixing
SMC – Sliding Mode Control CFG  (integrated into CWM hook)
ComfyUI Custom Node

References:
  DCW: "Elucidating the SNR-t Bias of Diffusion Probabilistic Models"
       Yu et al., arXiv:2604.16044v1 (2026)

  SMC: "CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance"
       Wang et al., CVPR 2026 / arXiv:2603.03281

─────────────────────────────────────────────────────────────────────────────
Architecture overview
─────────────────────────────────────────────────────────────────────────────

  sampler_cfg_function  (CWM hook):

    Standard path  (SMC off, alpha_l = alpha_h = 0):
        output = uncond + w · (cond − uncond)

    CWM only  (SMC off):
        e      = cond − uncond
        DWT(e) → LL, LH, HL, HH
        output = uncond + IDWT(w_LL·LL, w_mid·LH, w_mid·HL, w_HH·HH)

    SMC only  (alpha_l = alpha_h = 0):
        e      = cond − uncond
        s      = (e − e_prev) + λ · e_prev
        s_sm   = Gaussian(s)                      smooth spatial spikes
        τ      = mean|s_sm|                       adaptive scale
        e*     = e − k · tanh(s_sm / τ)          proportional switching
        output = uncond + w · e*

    SMC + CWM  (full integration — the new combined path):
        e      = cond − uncond
        s      = (e − e_prev) + λ · e_prev
        s_sm   = Gaussian(s)
        τ      = mean|s_sm|
        e*     = e − k · tanh(s_sm / τ)
        e_prev ← e*
        DWT(e*) → LL*, LH*, HL*, HH*
        output = uncond + IDWT(w_LL·LL*, w_mid·LH*, w_mid·HL*, w_HH·HH*)

    Rationale for SMC → CWM order:
      SMC corrects the *quality* of the guidance error (reduces oscillation,
      improves semantic alignment).  CWM then distributes the already-corrected
      error across frequency bands.  Applying wavelet mixing to a noisy error
      first and then trying to correct it would mis-attribute the oscillation
      energy to the wrong frequency bands, so SMC must precede CWM.

  sampler_post_cfg_function  (DCW hook):
    Always safe to chain; corrects x0_pred after CFG guidance is complete.

─────────────────────────────────────────────────────────────────────────────
SMC hyperparameter presets  (λ, k) — paper Sec 7.3 / Table 3
─────────────────────────────────────────────────────────────────────────────
  SD1.5 / SD2   λ=5, k=0.10
  SDXL          λ=5, k=0.10
  SD3 / SD3.5   λ=6, k=0.10  (grid-searched)
  Flux          λ=6, k=0.70  (grid-searched)
  Qwen-Image    λ=6, k=0.10  (grid-searched)
  Cosmos / Wan  λ=6, k=0.20  (conservative)
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SMC presets
# ─────────────────────────────────────────────────────────────────────────────

_SMC_PRESETS: dict[str, tuple[float, float]] = {
    "SD1.5 / SD2":  (5.0, 0.10),
    "SDXL":         (5.0, 0.10),
    "SD3 / SD3.5":  (6.0, 0.10),
    "Flux":         (6.0, 0.70),
    "Qwen-Image":   (6.0, 0.10),
    "Cosmos / Wan": (6.0, 0.20),
    "Custom":       (6.0, 0.10),
}

SMC_PRESET_NAMES: list[str] = ["Off", "Auto"] + list(_SMC_PRESETS.keys())

# State key for cross-step e_prev persistence inside model_options.
# model_options is a fresh dict per KSampler run → auto-resets between runs.
_SMC_STATE_KEY = "_dcw_smc_state"

# SMC minimum scale guard (prevents division by zero in tanh normalisation)
_SMC_CLAMP_MIN   = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Model family auto-detection for SMC preset
# ─────────────────────────────────────────────────────────────────────────────

def _detect_smc_preset(model) -> str:
    """Infer the best SMC preset from the model object's class name / attributes."""
    try:
        inner = model.model
        cls = type(inner).__name__.lower()

        if "flux"    in cls:                        return "Flux"
        if "cosmos"  in cls or "predict2" in cls:   return "Cosmos / Wan"
        if "wan"     in cls:                        return "Cosmos / Wan"
        if "anima"   in cls:                        return "Cosmos / Wan"
        if "sd3"     in cls or "mmdit"    in cls:   return "SD3 / SD3.5"
        if "sdxl"    in cls:                        return "SDXL"
        if "qwen"    in cls:                        return "Qwen-Image"

        mt = str(getattr(inner, "model_type", "")).lower()
        if "flow"   in mt: return "SD3 / SD3.5"
        if "v_pred" in mt: return "SD1.5 / SD2"

        diff = getattr(inner, "diffusion_model", None)
        if diff is not None:
            dm = type(diff).__name__.lower()
            if "flux"   in dm:                      return "Flux"
            if "cosmos" in dm or "wan" in dm:       return "Cosmos / Wan"
            if "mmdit"  in dm or "sd3" in dm:       return "SD3 / SD3.5"
            if "sdxl"   in dm:                      return "SDXL"
    except Exception:
        pass
    return "SD1.5 / SD2"


# ─────────────────────────────────────────────────────────────────────────────
# Haar Wavelet Utilities  (pure PyTorch, zero extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _pad_even(x):
    """
    Pad H and W to even numbers if necessary (Haar DWT requires even dims).
    Returns (padded_tensor, (original_H, original_W)).

    Cosmos VAE (used by Anima) compresses space by 8×, so e.g. resolution
    1208 → latent H 151 (odd), which triggers this path on a 5-D video tensor.
    """
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = H % 2, W % 2
    if ph or pw:
        extra = max(0, x.dim() - 4)
        pad = (0, pw, 0, ph) + (0, 0) * extra
        x = F.pad(x, pad, mode="reflect")
    return x, (H, W)


def haar_dwt2d(x):
    """
    2-D Haar Discrete Wavelet Transform.

    Input : (B, C, H, W) — H and W must be even
    Output: four subbands, each (B, C, H//2, W//2)
        LL – low-low   : coarse structure / global energy
        LH – low-high  : horizontal edges
        HL – high-low  : vertical edges
        HH – high-high : diagonal / fine texture
    """
    L  = (x[..., 0::2, :] + x[..., 1::2, :]) * 0.5
    Hi = (x[..., 0::2, :] - x[..., 1::2, :]) * 0.5

    LL = (L[...,  0::2] + L[...,  1::2]) * 0.5
    LH = (L[...,  0::2] - L[...,  1::2]) * 0.5
    HL = (Hi[..., 0::2] + Hi[..., 1::2]) * 0.5
    HH = (Hi[..., 0::2] - Hi[..., 1::2]) * 0.5

    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """
    2-D Haar Inverse Discrete Wavelet Transform.

    Handles arbitrary leading dimensions:
        4-D image : (B, C,    h, w) →  (B, C,    2h, 2w)
        5-D video : (B, C, T, h, w) →  (B, C, T, 2h, 2w)
    """
    *leading, h, w = LL.shape
    dev, dt = LL.device, LL.dtype

    L  = torch.empty(*leading, h, w * 2, device=dev, dtype=dt)
    Hi = torch.empty(*leading, h, w * 2, device=dev, dtype=dt)
    L[...,  0::2] = LL + LH
    L[...,  1::2] = LL - LH
    Hi[..., 0::2] = HL + HH
    Hi[..., 1::2] = HL - HH

    out = torch.empty(*leading, h * 2, w * 2, device=dev, dtype=dt)
    out[..., 0::2, :] = L + Hi
    out[..., 1::2, :] = L - Hi

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

_FP8_DTYPES = {
    dt for name in ("float8_e4m3fn", "float8_e5m2",
                    "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dt := getattr(torch, name, None)) is not None
}


def _safe_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp8 → bfloat16 for arithmetic safety; everything else unchanged."""
    if dtype in _FP8_DTYPES:
        return torch.bfloat16
    return dtype


def _sigma_norm(sigma, tensor: torch.Tensor, compute_dtype: torch.dtype):
    """
    σ_norm = σ / (σ + 1)  →  [0, 1)

    Consistent normalisation across EDM / DDPM / Flow schedulers.
    Always computed in fp32 to avoid precision loss at small sigma.
    """
    if isinstance(sigma, torch.Tensor):
        s = sigma.float() / (sigma.float() + 1.0)
        if s.dim() == 0:
            pass
        elif s.dim() == 1:
            extra = tensor.dim() - 1
            s = s.view(-1, *([1] * extra))
        return s.to(dtype=compute_dtype, device=tensor.device)
    else:
        sigma_f = float(sigma)
        return sigma_f / (sigma_f + 1.0)


def _ch_energy_weight(x: torch.Tensor,
                      clamp_lo: float = 0.25,
                      clamp_hi: float = 4.0) -> torch.Tensor:
    """
    Per-channel energy weight for DCW correction.

    Motivation
    ----------
    In multi-channel VAEs (e.g. Cosmos 16-ch used by Anima), each channel
    encodes different semantic content — pose, accessory presence, lighting,
    etc.  Applying a uniform correction lambda across all channels treats a
    semantically-rich channel and a near-empty background channel identically.
    This causes small lambda changes (~0.01) to push high-energy semantic
    channels past decision boundaries, producing the discrete jumps (body
    angle, accessory appearing/disappearing) observed in practice.

    Formula
    -------
    For a subband tensor x of shape (B, C, ...) :

        energy_c   = mean_{spatial}( x_c² )          per channel
        weight_c   = energy_c / mean_c( energy_c )   normalise → mean=1
        weight_c   = clamp( weight_c, lo, hi )        stability guard

    Properties
    ----------
    • Mean weight is always 1.0  →  total correction energy is unchanged.
    • High-energy (semantically active) channels get stronger correction;
      low-energy channels get weaker correction.
    • In early steps HH_x is dominated by noise (uniform across channels)
      → weights ≈ 1 → effectively no per-channel bias when reference is
      meaningless, which is the correct behaviour.
    • Clamping [0.25, 4.0] limits the range to 1/4× – 4×; prevents a
      single loud channel from monopolising or a quiet channel from being
      corrected into instability.

    Parameters
    ----------
    x        : reference subband tensor  (B, C, ...)
    clamp_lo : minimum weight (default 0.25)
    clamp_hi : maximum weight (default 4.0)

    Returns tensor of same shape, broadcastable with x.
    """
    spatial = tuple(range(2, x.dim()))                              # all dims after C
    energy  = x.float().pow(2).mean(dim=spatial, keepdim=True)     # (B, C, 1...)
    mean_e  = energy.mean(dim=1, keepdim=True).clamp(min=1e-8)     # (B, 1, 1...)
    return (energy / mean_e).clamp(clamp_lo, clamp_hi).to(x.dtype)


# Gaussian kernel parameters for SMC smooth switching
_SMC_BLUR_KERNEL_SIZE = 5
_SMC_BLUR_SIGMA       = 1.0


def _gaussian_blur_2d(x: torch.Tensor,
                      kernel_size: int = _SMC_BLUR_KERNEL_SIZE,
                      sigma: float     = _SMC_BLUR_SIGMA) -> torch.Tensor:
    """
    Separable 2-D Gaussian blur applied to the last two spatial dimensions.

    Handles arbitrary leading dimensions (4-D image, 5-D video latents).
    The blur removes high-frequency spatial spikes from the SMC sliding
    surface s before tanh switching, preventing checkerboard artifacts
    that arise from element-wise sign(s).

    Implementation:
        1-D Gaussian kernel built analytically (no scipy dependency).
        Applied as two successive 1-D conv ops (rows then cols) for O(k)
        cost rather than O(k²) from a full 2-D kernel.
        Reflect padding preserves border behaviour without edge discontinuities.

    Parameters
    ----------
    x           : input tensor  (..., H, W)  — expected float32 from SMC upcast
    kernel_size : must be odd; defaults to 5
    sigma       : Gaussian σ in pixels; defaults to 1.0
    """
    # Build 1-D Gaussian kernel (CPU, then move to x.device)
    half  = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - half   # [-2,-1,0,1,2]
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = (kernel_1d / kernel_1d.sum()).to(device=x.device, dtype=x.dtype)

    # Reshape input to (N, C, H, W) for F.conv2d, preserving leading dims
    orig_shape = x.shape
    if x.dim() == 4:
        t = x
    elif x.dim() > 4:
        # Fold all leading dims (B, C, T, ...) into batch dim
        t = x.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
    else:
        t = x.unsqueeze(0)

    C = t.shape[1] if t.dim() == 4 else 1

    # Kernels: (out_ch, in_ch/groups, 1, k) and (out_ch, in_ch/groups, k, 1)
    k_row = kernel_1d.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
    k_col = kernel_1d.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)

    pad = kernel_size // 2
    t = F.pad(t, (pad, pad, 0,   0), mode="reflect")
    t = F.conv2d(t, k_row, groups=C)
    t = F.pad(t, (0,   0, pad, pad), mode="reflect")
    t = F.conv2d(t, k_col, groups=C)

    return t.reshape(orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
# DCW Core  (post-CFG: corrects x0_pred)
# ─────────────────────────────────────────────────────────────────────────────

def apply_dcw(denoised: torch.Tensor,
              x_t:      torch.Tensor,
              sigma,
              lambda_l: float,
              lambda_h: float) -> torch.Tensor:
    """
    Apply Differential Correction in Wavelet domain to x0_pred.

    Correction formula per frequency band f:
        corrected_f = denoised_f + λ_f(t) · w_ch_f · (x_t_f − denoised_f)

    Band timing
    -----------
    LL  (global structure)    λ_l(t) = lambda_l · σ_norm         early steps
    LH  (horiz. edges)  ┐    λ_m(t) = (λ_l + λ_h) / 2          mid steps
    HL  (vert. edges)   ┘    (linear interpolation of LL and HH schedules)
    HH  (fine texture)        λ_h(t) = lambda_h · (1 − σ_norm)   late steps

    LH and HL represent directional edge information that forms between
    global structure (LL) and fine texture (HH) in the coarse-to-fine
    denoising process.  Giving them the same timing as HH (late-only)
    was incorrect; the interpolated schedule is more physically accurate.

    Per-channel energy weighting
    ----------------------------
    w_ch_f = _ch_energy_weight(x_t_f)

    Channels with higher signal energy receive proportionally stronger
    correction.  This prevents small lambda changes from causing discrete
    semantic jumps (body pose, accessory presence) in multi-channel VAEs
    such as Cosmos 16-ch (Anima), where different channels encode different
    semantic concepts at different energy levels.

    When x_t_f is dominated by noise (early steps, HH band), the energy
    is approximately uniform across channels → weights ≈ 1 → no artificial
    bias is introduced where the reference signal is not yet meaningful.
    """
    if lambda_l == 0.0 and lambda_h == 0.0:
        return denoised

    orig_dtype    = denoised.dtype
    compute_dtype = _safe_compute_dtype(orig_dtype)
    need_cast     = (compute_dtype != orig_dtype)

    if need_cast:
        denoised = denoised.to(dtype=compute_dtype)

    s = _sigma_norm(sigma, denoised, compute_dtype)

    lam_l   = lambda_l * s
    lam_h   = lambda_h * (1.0 - s)
    lam_mid = (lam_l + lam_h) * 0.5          # LH / HL: mid-step timing

    dn_p, (H, W) = _pad_even(denoised)
    xt_p, _      = _pad_even(x_t.to(dtype=compute_dtype, device=denoised.device))

    LL_d, LH_d, HL_d, HH_d = haar_dwt2d(dn_p)
    LL_x, LH_x, HL_x, HH_x = haar_dwt2d(xt_p)

    # Per-channel energy weights derived from x_t subbands
    cw_LL = _ch_energy_weight(LL_x)
    cw_LH = _ch_energy_weight(LH_x)
    cw_HL = _ch_energy_weight(HL_x)
    cw_HH = _ch_energy_weight(HH_x)

    LL_c = LL_d + lam_l   * cw_LL * (LL_x - LL_d)
    LH_c = LH_d + lam_mid * cw_LH * (LH_x - LH_d)
    HL_c = HL_d + lam_mid * cw_HL * (HL_x - HL_d)
    HH_c = HH_d + lam_h   * cw_HH * (HH_x - HH_d)

    out = haar_idwt2d(LL_c, LH_c, HL_c, HH_c)
    out = out[..., :H, :W]

    if need_cast:
        out = out.to(dtype=orig_dtype)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Combined CFG hook: SMC + CWM
# ─────────────────────────────────────────────────────────────────────────────

def apply_cfg_combined(
    cond:       torch.Tensor,
    uncond:     torch.Tensor,
    sigma,
    cfg_scale:  float,
    # CWM params
    alpha_l:    float,
    alpha_h:    float,
    # SMC params  (lambda_val=0 / k=0 → SMC disabled)
    smc_lambda: float,
    smc_k:      float,
    # SMC cross-step state container (mutated in-place)
    smc_state:  dict,
    # fp32 upcast for SMC (prevents NaN with fp16/bf16 in edge cases)
    fp32_smc:   bool = True,
) -> torch.Tensor:
    """
    Unified CFG computation.  Execution order:

        1. Compute raw guidance error  e = cond − uncond
        2. [Optional] SMC correction:  e* = SMC(e, e_prev, λ, k)
        3. [Optional] CWM scaling:     apply per-band adaptive weights
        4. Reconstruct: uncond + guided_error

    When both SMC and CWM are inactive, reduces exactly to standard CFG.
    When only SMC is active (alpha_l = alpha_h = 0), no wavelet ops run.
    When only CWM is active (smc_lambda = 0 or smc_k = 0), no SMC state ops run.
    """
    smc_active = (smc_lambda != 0.0 and smc_k != 0.0)
    cwm_active = (alpha_l != 0.0 or alpha_h != 0.0)

    # ── Fast path: pure standard CFG ──────────────────────────────────────────
    if not smc_active and not cwm_active:
        return uncond + cfg_scale * (cond - uncond)

    orig_dtype    = cond.dtype
    compute_dtype = _safe_compute_dtype(orig_dtype)
    need_cast     = (compute_dtype != orig_dtype)

    # SMC upcast target
    smc_dtype = torch.float32 if fp32_smc else compute_dtype

    if need_cast:
        cond_w   = cond.to(dtype=compute_dtype)
        uncond_w = uncond.to(dtype=compute_dtype)
    else:
        cond_w, uncond_w = cond, uncond

    # ── Step 1: raw guidance error ────────────────────────────────────────────
    e = torch.nan_to_num(cond_w - uncond_w, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Step 2: SMC correction ────────────────────────────────────────────────
    if smc_active:
        e_smc = e.to(dtype=smc_dtype)

        e_prev_raw = smc_state.get("e_prev")
        if e_prev_raw is None:
            # First step: initialise e_prev = e  (Algorithm 1, line 7-9)
            e_prev = e_smc.detach().clone()
        else:
            e_prev = torch.nan_to_num(
                e_prev_raw.to(device=e_smc.device, dtype=smc_dtype),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

        # Sliding surface:  s = (e - e_prev) + λ · e_prev
        s = torch.nan_to_num(
            (e_smc - e_prev) + smc_lambda * e_prev,
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        # Switching control: Gaussian smoothing → tanh  (replaces sign(s))
        #
        # Original paper uses:  u_sw = −k · sign(s)
        # This produces element-wise ±k — a hard checkerboard pattern that
        # causes spatial spike artifacts and discrete semantic jumps in
        # multi-channel VAEs (Cosmos 16-ch, Anima).
        #
        # Replacement (README Sec "원본 논문 대비 개선"):
        #   s_smooth = Gaussian(s)          — removes spatial spikes
        #   τ        = mean|s_smooth|       — step/model-agnostic adaptive scale
        #   Δe       = −k · tanh(s_smooth / τ)
        #
        # Properties vs sign(s):
        #   • Spatially continuous — no checkerboard ±1 pattern
        #   • Proportional to |s| — small surface → weak correction
        #   • Bounded ∈ (−k, +k) — Lyapunov stability preserved
        #   • τ normalisation makes k model/step invariant
        s_smooth = _gaussian_blur_2d(s)
        reduce_dims = tuple(range(1, s_smooth.ndim))
        tau     = s_smooth.abs().mean(dim=reduce_dims, keepdim=True).clamp(min=_SMC_CLAMP_MIN)
        delta_e = -smc_k * torch.tanh(s_smooth / tau)

        e_star = torch.nan_to_num(e_smc + delta_e, nan=0.0, posinf=0.0, neginf=0.0)

        # Update cross-step state with corrected error
        smc_state["e_prev"] = e_star.detach().clone()

        # Cast corrected error back to compute_dtype for subsequent ops
        e = e_star.to(dtype=compute_dtype)

    # ── Step 3: CWM per-band scaling ──────────────────────────────────────────
    if cwm_active:
        w = float(cfg_scale)
        s_n = _sigma_norm(sigma, e, compute_dtype)

        w_LL  = w * (1.0 + float(alpha_l) * s_n)
        w_HH  = w * (1.0 + float(alpha_h) * (1.0 - s_n))
        # Geometric mean for LH/HL: physically correct for a band that lies
        # between LL and HH in the coarse-to-fine hierarchy.
        # AM ≥ GM always, so arithmetic mean systematically over-weights the
        # mid-band; geometric mean keeps energy scaling consistent with the
        # log-linear interpolation implicit in wavelet decomposition.
        w_mid = (w_LL * w_HH) ** 0.5

        e_p, (H, W) = _pad_even(e)
        LL_e, LH_e, HL_e, HH_e = haar_dwt2d(e_p)

        LL_g = w_LL  * LL_e
        LH_g = w_mid * LH_e
        HL_g = w_mid * HL_e
        HH_g = w_HH  * HH_e

        e_guided = haar_idwt2d(LL_g, LH_g, HL_g, HH_g)
        e_guided = e_guided[..., :H, :W]

        out = uncond_w + e_guided
    else:
        # SMC only path — no wavelet overhead
        out = uncond_w + float(cfg_scale) * e

    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if need_cast:
        out = out.to(dtype=orig_dtype)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node Definition
# ─────────────────────────────────────────────────────────────────────────────

class DCWModelPatch:
    """
    DCW + CWM + SMC Model Patch node.

    DCW (sampler_post_cfg_function):
        Corrects x0_pred in the wavelet domain to mitigate SNR-t bias.
        Applied after CFG combination, every denoising step.
        Always safe to chain with other post-cfg hooks.

    CWM (sampler_cfg_function):
        Replaces scalar CFG with per-band adaptive guidance scaling.
        Low-freq bands (structure/layout) are boosted in early steps;
        high-freq bands (edges/texture) are boosted in late steps.
        alpha_l = alpha_h = 0 → identical to standard CFG.

    SMC (integrated into CWM hook):
        Sliding Mode Control correction of the guidance error prior to
        wavelet decomposition.  Reduces oscillation in the guidance
        trajectory and improves semantic alignment.
        smc_preset = Off → SMC disabled entirely.
        When CWM is also enabled, SMC runs first (corrects error quality),
        then CWM distributes the corrected error across frequency bands.
        When CWM is disabled, SMC runs as a pure CFG replacement with no
        wavelet overhead.

    Compatibility:
        DCW can co-exist with any other sampler_cfg_function node because
        it uses a different hook (sampler_post_cfg_function).
        CWM/SMC use sampler_cfg_function.  If another node has already
        registered that hook before this node runs, CWM/SMC are skipped
        automatically with a console warning.
    """

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "patch"
    CATEGORY      = "model_patches"
    DESCRIPTION   = (
        "DCW – Differential Correction in Wavelet domain (SNR-t bias correction).\n"
        "CWM – CFG Wavelet Mixing (adaptive per-band guidance scale).\n"
        "SMC – Sliding Mode Control CFG (integrated into CWM, corrects guidance error).\n"
        "All three are training-free and plug-and-play."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),

                # ── DCW parameters ────────────────────────────────────────────
                "lambda_l": ("FLOAT", {
                    "default": 0.05,
                    "min":    -0.5,
                    "max":     0.5,
                    "step":    0.005,
                    "round":   0.001,
                    "tooltip": (
                        "[DCW] Low-frequency correction strength (x0_pred).\n"
                        "Positive: push x0_pred toward x_t (more signal).\n"
                        "Negative: push away from x_t (less signal).\n"
                        "Active mainly in EARLY denoising steps.\n"
                        "Start: 0.04–0.07 for DDPM/EDM; ~2× for Flux/Flow.\n"
                        "Set 0.0 to disable DCW low-freq correction."
                    ),
                }),

                "lambda_h": ("FLOAT", {
                    "default": 0.01,
                    "min":    -0.3,
                    "max":     0.3,
                    "step":    0.001,
                    "round":   0.001,
                    "tooltip": (
                        "[DCW] High-frequency correction strength (x0_pred).\n"
                        "Positive: push x0_pred toward x_t (more detail).\n"
                        "Negative: push away from x_t (less detail).\n"
                        "Active mainly in LATE denoising steps.\n"
                        "Start: 0.008–0.015 for DDPM/EDM; ~2× for Flux/Flow.\n"
                        "Set 0.0 to disable DCW high-freq correction."
                    ),
                }),

                "dcw_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[DCW] Toggle SNR-t bias correction on/off.",
                }),

                # ── CWM parameters ────────────────────────────────────────────
                "alpha_l": ("FLOAT", {
                    "default": 0.0,
                    "min":    -1.0,
                    "max":     2.0,
                    "step":    0.01,
                    "round":   0.001,
                    "tooltip": (
                        "[CWM] Low-frequency CFG boost strength.\n"
                        "Scales the LL guidance band relative to base CFG scale.\n"
                        "Positive: amplify coarse structure/composition guidance in early steps.\n"
                        "Negative: suppress low-freq guidance.\n"
                        "0.0 = standard CFG for this band.\n"
                        "Suggested start: 0.1–0.3. Flow models (Flux): try 0.2–0.5."
                    ),
                }),

                "alpha_h": ("FLOAT", {
                    "default": 0.0,
                    "min":    -1.0,
                    "max":     2.0,
                    "step":    0.01,
                    "round":   0.001,
                    "tooltip": (
                        "[CWM] High-frequency CFG boost strength.\n"
                        "Scales the HH guidance band relative to base CFG scale.\n"
                        "Positive: amplify edge/texture/detail guidance in late steps.\n"
                        "Negative: suppress high-freq guidance.\n"
                        "0.0 = standard CFG for this band.\n"
                        "Suggested start: 0.1–0.2. Excessive values cause over-sharpening."
                    ),
                }),

                "cwm_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[CWM] Toggle CFG Wavelet Mixing on/off.",
                }),

                # ── SMC parameters ────────────────────────────────────────────
                "smc_preset": (
                    SMC_PRESET_NAMES,
                    {
                        "default": "Off",
                        "tooltip": (
                            "[SMC] Sliding Mode Control CFG preset.\n"
                            "'Off'   — SMC disabled entirely.\n"
                            "'Auto'  — auto-detect model family, apply paper defaults.\n"
                            "Named presets — paper-verified (λ, k) per model family.\n"
                            "'Custom' — use smc_lambda / smc_k sliders below.\n\n"
                            "SMC runs inside the CWM hook (sampler_cfg_function).\n"
                            "It corrects guidance error quality BEFORE CWM distributes\n"
                            "it across frequency bands.\n"
                            "If cwm_enabled=False, SMC still activates the cfg hook\n"
                            "and runs without wavelet overhead (pure SMC-CFG mode)."
                        ),
                    },
                ),

                "smc_lambda": ("FLOAT", {
                    "default": 6.0,
                    "min":     0.5,
                    "max":    30.0,
                    "step":    0.1,
                    "tooltip": (
                        "[SMC] Shape parameter λ of the sliding mode surface.\n"
                        "Only used when smc_preset = Custom.\n"
                        "Paper recommends 2–8.  Extreme values distort the manifold."
                    ),
                }),

                "smc_k": ("FLOAT", {
                    "default": 0.1,
                    "min":     0.0,
                    "max":     5.0,
                    "step":    0.01,
                    "tooltip": (
                        "[SMC] Switching gain k.\n"
                        "Only used when smc_preset = Custom.\n"
                        "Low k → better realism, weaker text alignment.\n"
                        "High k → stronger text alignment, risk of chattering."
                    ),
                }),
            }
        }

    # ── Patch method ──────────────────────────────────────────────────────────

    def patch(
        self,
        model,
        lambda_l:    float,
        lambda_h:    float,
        dcw_enabled: bool,
        alpha_l:     float,
        alpha_h:     float,
        cwm_enabled: bool,
        smc_preset:  str,
        smc_lambda:  float,
        smc_k:       float,
    ):
        # ── Resolve SMC hyperparameters ───────────────────────────────────────
        smc_on = (smc_preset != "Off")

        if smc_on:
            if smc_preset == "Auto":
                detected         = _detect_smc_preset(model)
                resolved_lambda, resolved_k = _SMC_PRESETS[detected]
                print(f"[DCW/SMC] Auto → '{detected}'  λ={resolved_lambda}  k={resolved_k}")
            elif smc_preset == "Custom":
                resolved_lambda, resolved_k = smc_lambda, smc_k
                print(f"[DCW/SMC] Custom  λ={resolved_lambda}  k={resolved_k}")
            else:
                resolved_lambda, resolved_k = _SMC_PRESETS[smc_preset]
                print(f"[DCW/SMC] '{smc_preset}'  λ={resolved_lambda}  k={resolved_k}")
        else:
            resolved_lambda = resolved_k = 0.0

        # ── Determine active features ─────────────────────────────────────────
        dcw_active = dcw_enabled and (lambda_l != 0.0 or lambda_h != 0.0)
        # cfg hook is needed if CWM alpha is non-zero OR SMC is on
        cwm_alpha_active = cwm_enabled and (alpha_l != 0.0 or alpha_h != 0.0)
        cfg_hook_needed  = cwm_alpha_active or smc_on

        if not dcw_active and not cfg_hook_needed:
            return (model,)

        m = model.clone()
        m.model_options = {**m.model_options}

        # ── CFG hook: CWM + (optional) SMC ───────────────────────────────────
        if cfg_hook_needed:
            if "sampler_cfg_function" in m.model_options:
                print(
                    "[DCW/CWM/SMC] Warning: sampler_cfg_function already registered "
                    "by another node.  CWM and SMC skipped to avoid conflict.\n"
                    "  → Place this node earlier in the pipeline, or disable the "
                    "conflicting node's CFG hook."
                )
            else:
                # Capture parameters and mutable SMC state in closure
                _al         = alpha_l  if cwm_alpha_active else 0.0
                _ah         = alpha_h  if cwm_alpha_active else 0.0
                _smc_lambda = resolved_lambda
                _smc_k      = resolved_k
                # smc_state dict is mutated per-step to carry e_prev.
                # model_options is a fresh dict per KSampler run, but the
                # closure itself persists across the node's lifetime; we
                # embed the state inside model_options so it resets correctly.
                _model_options_ref = m.model_options

                def cfg_hook(args: dict) -> torch.Tensor:
                    cond_t     = args.get("cond")
                    uncond_t   = args.get("uncond")
                    cond_scale = args.get("cond_scale", 7.0)
                    sigma      = args.get("sigma")

                    if cond_t is None or uncond_t is None:
                        return uncond_t + float(cond_scale) * (cond_t - uncond_t)

                    # Retrieve or create per-run SMC state
                    opts      = args.get("model_options", _model_options_ref)
                    smc_state = opts.setdefault(_SMC_STATE_KEY, {})

                    try:
                        return apply_cfg_combined(
                            cond_t, uncond_t, sigma,
                            float(cond_scale),
                            _al, _ah,
                            _smc_lambda, _smc_k,
                            smc_state,
                            fp32_smc=True,
                        )
                    except Exception as exc:
                        print(f"[CWM/SMC] Warning: skipped at this step – {exc}")
                        return uncond_t + float(cond_scale) * (cond_t - uncond_t)

                m.model_options["sampler_cfg_function"]      = cfg_hook
                m.model_options["disable_cfg1_optimization"] = True

        # ── Post-CFG hook: DCW ────────────────────────────────────────────────
        if dcw_active:
            existing_fns = list(m.model_options.get("sampler_post_cfg_function", []))
            _ll = lambda_l
            _lh = lambda_h

            def dcw_post_cfg(args: dict) -> torch.Tensor:
                denoised = args.get("denoised")
                x_t      = args.get("input")
                sigma    = args.get("sigma")

                if denoised is None or x_t is None or sigma is None:
                    return denoised

                try:
                    return apply_dcw(denoised, x_t, sigma, _ll, _lh)
                except Exception as exc:
                    print(f"[DCW] Warning: correction skipped at this step – {exc}")
                    return denoised

            existing_fns.append(dcw_post_cfg)
            m.model_options["sampler_post_cfg_function"] = existing_fns

        return (m,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DCWModelPatch": DCWModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DCWModelPatch": "DCW + CWM + SMC Model Patch",
}
