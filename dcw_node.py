"""
DCW – Differential Correction in Wavelet domain
ComfyUI Custom Node

Mitigates SNR-t bias in Diffusion Probabilistic Models.
Based on: "Elucidating the SNR-t Bias of Diffusion Probabilistic Models"
          arXiv:2604.16044v1 (Yu et al., 2026)

Implementation note: applied as a post-CFG hook on x0_pred,
not directly on x_{t-1} as in the paper. See README for details.
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Haar Wavelet Utilities  (pure PyTorch, zero extra dependencies)
# ─────────────────────────────────────────────────────────────

def _pad_even(x):
    """
    Pad H and W to even numbers if necessary (Haar DWT requires even dims).
    Returns (padded_tensor, (original_H, original_W)).

    PyTorch reflect padding rules (elements in tuple = 2 × number of padded dims,
    specified from last dim to first):
        4-D tensor → 4-element tuple  (pads W and H)
        5-D tensor → 6-element tuple  (pads W, H, and T with 0)

    Cosmos VAE (used by Anima) compresses space by 8x, so e.g. resolution
    1208 -> latent H 151 (odd), which triggers this path on a 5-D video tensor.
    """
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = H % 2, W % 2
    if ph or pw:
        # (0, pw, 0, ph) covers W and H
        # prepend (0, 0) for every extra leading dim beyond 4-D
        extra = max(0, x.dim() - 4)
        pad = (0, pw, 0, ph) + (0, 0) * extra
        x = F.pad(x, pad, mode="reflect")
    return x, (H, W)


def haar_dwt2d(x):
    """
    2-D Haar Discrete Wavelet Transform.

    Input : (B, C, H, W)  – H and W must be even
    Output: four subbands, each (B, C, H//2, W//2)
        LL – low-low   : coarse structure / global energy
        LH – low-high  : horizontal edges
        HL – high-low  : vertical edges
        HH – high-high : diagonal / fine texture
    """
    # 1-D DWT along H axis
    L  = (x[..., 0::2, :] + x[..., 1::2, :]) * 0.5
    Hi = (x[..., 0::2, :] - x[..., 1::2, :]) * 0.5

    # 1-D DWT along W axis
    LL = (L[...,  0::2] + L[...,  1::2]) * 0.5
    LH = (L[...,  0::2] - L[...,  1::2]) * 0.5
    HL = (Hi[..., 0::2] + Hi[..., 1::2]) * 0.5
    HH = (Hi[..., 0::2] - Hi[..., 1::2]) * 0.5

    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """
    2-D Haar Inverse Discrete Wavelet Transform.

    Handles arbitrary leading dimensions:
        4-D image  : (B, C,    h, w)  →  (B, C,    2h, 2w)
        5-D video  : (B, C, T, h, w)  →  (B, C, T, 2h, 2w)
    """
    *leading, h, w = LL.shape   # unpack any number of leading dims
    dev, dt = LL.device, LL.dtype

    # Inverse DWT along W axis
    L  = torch.empty(*leading, h, w * 2, device=dev, dtype=dt)
    Hi = torch.empty(*leading, h, w * 2, device=dev, dtype=dt)
    L[...,  0::2] = LL + LH
    L[...,  1::2] = LL - LH
    Hi[..., 0::2] = HL + HH
    Hi[..., 1::2] = HL - HH

    # Inverse DWT along H axis
    out = torch.empty(*leading, h * 2, w * 2, device=dev, dtype=dt)
    out[..., 0::2, :] = L + Hi
    out[..., 1::2, :] = L - Hi

    return out


# ─────────────────────────────────────────────────────────────
# DCW Core Correction
# ─────────────────────────────────────────────────────────────

# fp8 dtypes introduced in PyTorch 2.1 – collect whichever exist in this build
_FP8_DTYPES = {
    dt for name in ("float8_e4m3fn", "float8_e5m2",
                    "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dt := getattr(torch, name, None)) is not None
}


def _safe_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Return a dtype that supports standard arithmetic.

    fp8 variants  → bfloat16  (best perf on modern GPUs; no inf)
    everything else is already arithmetic-safe, return as-is.
    """
    if dtype in _FP8_DTYPES:
        return torch.bfloat16
    return dtype


def apply_dcw(denoised: torch.Tensor,
              x_t:      torch.Tensor,
              sigma,
              lambda_l: float,
              lambda_h: float) -> torch.Tensor:
    """
    Apply Differential Correction in Wavelet domain to x0_pred.

    Args:
        denoised : x0_pred from the model,  shape (B, C, H, W)
        x_t      : noisy latent at current step, shape (B, C, H, W)
        sigma    : current denoising sigma (scalar tensor or float)
        lambda_l : low-frequency  correction strength (hyperparameter)
        lambda_h : high-frequency correction strength (hyperparameter)

    Returns:
        Corrected x0_pred, same shape as denoised.

    Correction formula per frequency band f:
        corrected_f = denoised_f  +  λ_f(t) · (x_t_f − denoised_f)

    where the correction direction (x_t_f − denoised_f) contains a
    positive signal component (1 − γ_t) · x0 that helps restore the
    underestimated signal caused by the SNR-t bias.

    Dynamic weights (σ_norm = σ / (σ + 1)  maps any σ to [0, 1)):
        λ_l(t) = lambda_l · σ_norm          ← larger at early steps
        λ_h(t) = lambda_h · (1 − σ_norm)    ← larger at late steps

    Dtype handling:
        fp32 / fp16 / bf16 → computed in-dtype (no cast needed)
        fp8 (e4m3fn etc.)  → upcasted to bf16 for computation,
                             result cast back to original fp8 dtype
    """
    if lambda_l == 0.0 and lambda_h == 0.0:
        return denoised

    # ── Dtype safety ────────────────────────────────────────────
    # fp8 tensors cannot perform standard arithmetic directly.
    # We upcast to a safe compute dtype and cast back at the end.
    orig_dtype   = denoised.dtype
    compute_dtype = _safe_compute_dtype(orig_dtype)
    need_cast    = (compute_dtype != orig_dtype)

    if need_cast:
        denoised = denoised.to(dtype=compute_dtype)

    # ── Sigma normalisation ─────────────────────────────────────
    # Works across all schedulers / model families:
    #   EDM  (σ ∈ [0.002, 14.6])  →  σ_norm ∈ [0.002, 0.936]
    #   DDPM (σ small)            →  σ_norm ≈ σ  (≪1)
    #   Flow (σ ∈ [0, 1])         →  σ_norm ∈ [0, 0.5]
    # For flow-based models (Flux, Anima/Cosmos) the effective range
    # tops out at ~0.5; use 2× the lambda values as a starting point.
    if isinstance(sigma, torch.Tensor):
        # Always compute sigma normalisation in fp32 to avoid
        # precision loss when sigma is very small (late denoising steps)
        s = sigma.float() / (sigma.float() + 1.0)
        if s.dim() == 0:
            pass                           # scalar, broadcasts fine
        elif s.dim() == 1:
            # Reshape to (B, 1, 1, ..., 1) matching denoised.ndim
            extra = denoised.dim() - 1     # number of trailing 1s needed
            s = s.view(-1, *([1] * extra)) # works for 4D and 5D alike
        s = s.to(dtype=compute_dtype, device=denoised.device)
    else:
        sigma_f = float(sigma)
        s = sigma_f / (sigma_f + 1.0)     # plain Python float, no dtype issue

    lam_l = lambda_l * s             # low-freq  weight
    lam_h = lambda_h * (1.0 - s)    # high-freq weight

    # ── Pad to even H, W ────────────────────────────────────────
    dn_p, (H, W) = _pad_even(denoised)
    xt_p, _      = _pad_even(x_t.to(dtype=compute_dtype, device=denoised.device))

    # ── Wavelet decomposition ───────────────────────────────────
    LL_d, LH_d, HL_d, HH_d = haar_dwt2d(dn_p)
    LL_x, LH_x, HL_x, HH_x = haar_dwt2d(xt_p)

    # ── Per-band differential correction ───────────────────────
    # Low-frequency subband  (structure, coarse shape)
    LL_c = LL_d + lam_l * (LL_x - LL_d)

    # High-frequency subbands  (edges, texture, detail)
    LH_c = LH_d + lam_h * (LH_x - LH_d)
    HL_c = HL_d + lam_h * (HL_x - HL_d)
    HH_c = HH_d + lam_h * (HH_x - HH_d)

    # ── Reconstruct & crop ──────────────────────────────────────
    out = haar_idwt2d(LL_c, LH_c, HL_c, HH_c)
    out = out[..., :H, :W]

    # Cast back to original dtype if we upcasted (e.g. fp8 → bf16 → fp8)
    if need_cast:
        out = out.to(dtype=orig_dtype)

    return out


# ─────────────────────────────────────────────────────────────
# ComfyUI Node Definition
# ─────────────────────────────────────────────────────────────

class DCWModelPatch:
    """
    DCW Model Patch node.

    Registers a post-CFG hook on the model that applies DCW correction
    after every denoising step.  Fully compatible with any sampler
    (euler_a, dpmpp_2m, lcm, …) and any scheduler (normal, AYS, …).
    """

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "patch"
    CATEGORY      = "model_patches"
    DESCRIPTION   = (
        "DCW – Differential Correction in Wavelet domain.\n"
        "Mitigates SNR-t bias that degrades generation quality,\n"
        "especially at low step counts.  Training-free, plug-and-play."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),

                "lambda_l": ("FLOAT", {
                    "default": 0.05,
                    "min":     0.0,
                    "max":     0.5,
                    "step":    0.005,
                    "round":   0.001,
                    "tooltip": (
                        "Low-frequency correction strength.\n"
                        "Active mainly in EARLY denoising steps.\n"
                        "Corrects coarse structure / global composition.\n"
                        "Start around 0.04–0.07 for DDPM/EDM models.\n"
                        "For flow-based models (Flux, Anima/Cosmos), try 2×."
                    ),
                }),

                "lambda_h": ("FLOAT", {
                    "default": 0.01,
                    "min":     0.0,
                    "max":     0.3,
                    "step":    0.001,
                    "round":   0.001,
                    "tooltip": (
                        "High-frequency correction strength.\n"
                        "Active mainly in LATE denoising steps.\n"
                        "Corrects fine detail, edges, and texture.\n"
                        "Start around 0.008–0.015 for DDPM/EDM models.\n"
                        "For flow-based models (Flux, Anima/Cosmos), try 2×."
                    ),
                }),

                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Quickly toggle the correction on/off for A/B comparison.",
                }),
            }
        }

    # ── Patch method ─────────────────────────────────────────────

    def patch(self, model, lambda_l: float, lambda_h: float, enabled: bool):
        if not enabled or (lambda_l == 0.0 and lambda_h == 0.0):
            return (model,)

        m = model.clone()

        # Deep-copy the options dict so we don't mutate the original model
        m.model_options = {**m.model_options}
        existing_fns = list(m.model_options.get("sampler_post_cfg_function", []))

        # Capture lambda values for the closure
        _ll = lambda_l
        _lh = lambda_h

        def dcw_post_cfg(args: dict) -> torch.Tensor:
            """
            Post-CFG hook called by ComfyUI after every denoising step.

            args keys guaranteed by ComfyUI:
                "denoised" – x0_pred (CFG-combined clean prediction)
                "input"    – x_t (current noisy latent fed to the model)
                "sigma"    – current sigma value for this step
            """
            denoised = args.get("denoised")
            x_t      = args.get("input")
            sigma    = args.get("sigma")

            if denoised is None or x_t is None or sigma is None:
                return denoised  # missing context → skip silently

            try:
                return apply_dcw(denoised, x_t, sigma, _ll, _lh)
            except Exception as exc:
                # Never crash the sampling run; just warn and pass through
                print(f"[DCW] Warning: correction skipped at this step – {exc}")
                return denoised

        existing_fns.append(dcw_post_cfg)
        m.model_options["sampler_post_cfg_function"] = existing_fns
        return (m,)


# ─────────────────────────────────────────────────────────────
# ComfyUI Registration
# ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DCWModelPatch": DCWModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DCWModelPatch": "DCW Model Patch (SNR-t Bias Correction)",
}
