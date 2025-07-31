import torch


def bilinear_from_tokens_hw(tokens_hw_c: torch.Tensor,
                             x_cont: torch.Tensor,
                             y_cont: torch.Tensor,
                             align_corners: bool = False,
                             clamp: bool = True) -> torch.Tensor:
    """
    Bilinear interpolate from a (Hc, Wc, C) token grid using continuous coordinates
    on that grid (x_cont in [0..Wc-1], y_cont in [0..Hc-1]).
    Returns (N, C).
    """
    device = tokens_hw_c.device
    dtype = tokens_hw_c.dtype
    Hc, Wc, C = tokens_hw_c.shape

    if align_corners:
        # centers at integers; leave here for completeness, default is False
        xp = x_cont
        yp = y_cont
    else:
        xp = x_cont
        yp = y_cont

    if clamp:
        eps = 1e-6
        xp = torch.clamp(xp, 0.0, float(Wc - 1) - eps)
        yp = torch.clamp(yp, 0.0, float(Hc - 1) - eps)

    ix0 = torch.floor(xp).to(torch.long)
    iy0 = torch.floor(yp).to(torch.long)
    ix1 = torch.clamp(ix0 + 1, max=Wc - 1)
    iy1 = torch.clamp(iy0 + 1, max=Hc - 1)

    tx = (xp - ix0.to(dtype)).unsqueeze(1)
    ty = (yp - iy0.to(dtype)).unsqueeze(1)

    flat = tokens_hw_c.view(Hc * Wc, C)
    idx00 = iy0 * Wc + ix0
    idx10 = iy0 * Wc + ix1
    idx01 = iy1 * Wc + ix0
    idx11 = iy1 * Wc + ix1

    f00 = flat[idx00]
    f10 = flat[idx10]
    f01 = flat[idx01]
    f11 = flat[idx11]

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    feats = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11
    return feats
