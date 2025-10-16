# Winsorized Entropy–Depth-Gated Exponential

from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import torch

@merge_method
def wedge(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha: Parameter(float) = 0.5,
    tmin: Parameter(float) = 0.5,
    tmax: Parameter(float) = 3.5,
    tau_lo: Parameter(float) = 0.10,
    tau_hi: Parameter(float) = 0.60,
    winsor_k: Parameter(float) = 3.0,
    depth_scale: Parameter(float) = 1.0,
    lambda_ce: Parameter(float) = 0.6,
    t0: Parameter(float) = 1.0,
    trust_k: Parameter(float) = 3.0,
    eps: Parameter(float) = 1e-8,
) -> Return(Tensor):
    wa = torch.as_tensor(1.0 - float(alpha), device=a.device, dtype=a.dtype)
    wb = torch.as_tensor(float(alpha), device=a.device, dtype=a.dtype)

    def winsorize(x: Tensor) -> Tensor:
        if float(winsor_k) <= 0.0:
            return x
        flat = x.flatten()
        m = flat.median()
        mad = (flat - m).abs().median() + eps
        lo = m - float(winsor_k) * mad
        hi = m + float(winsor_k) * mad
        return x.clamp(lo, hi)

    a_ = winsorize(a)
    b_ = winsorize(b)

    af = a_.flatten()
    bf = b_.flatten()
    cos = (af @ bf) / ((af.norm() + eps) * (bf.norm() + eps))
    d_cos = 0.5 * (1.0 - cos.clamp(-1.0, 1.0))

    sa = float(t0) * a_.mean()
    sb = float(t0) * b_.mean()
    scores = torch.stack([sa, sb], dim=0)
    P = torch.softmax(scores, dim=0)
    H = -(P * (P + eps).log()).sum()
    d_H = 1.0 - H / torch.log(torch.tensor(2.0, device=a.device, dtype=a.dtype))

    d = float(lambda_ce) * d_cos + (1.0 - float(lambda_ce)) * d_H
    psi = ((d - float(tau_lo)) / (float(tau_hi) - float(tau_lo) + eps)).clamp(0.0, 1.0)
    psi = psi * float(depth_scale)
    t = float(tmin) + (float(tmax) - float(tmin)) * psi

    logw_a = torch.log(wa + eps)
    logw_b = torch.log(wb + eps)
    stacked = torch.stack([t * a_ + logw_a, t * b_ + logw_b], dim=0)
    y = torch.logsumexp(stacked, dim=0) / (t + eps)

    mu_a = a_.mean()
    mu_y = y.mean()
    std_a = a_.std().clamp_min(eps)
    std_y = y.std().clamp_min(eps)
    y = mu_a + (std_a / std_y) * (y - mu_y)

    flat_a = a_.flatten()
    med = flat_a.median()
    mad = (flat_a - med).abs().median() + eps
    r = float(trust_k) * mad
    y = a_ + (y - a_).clamp(-r, r)

    y = torch.where(torch.isfinite(y), y, a_)
    return y
