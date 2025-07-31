from typing import Dict, Optional

import torch
import torch.nn.functional as F


def info_nce_pairwise(
    z_student: torch.Tensor,   # (N,D) assumed L2-normalized or not; we L2-normalize inside
    z_teacher: torch.Tensor,   # (N,D) teacher for the same points (1:1)
    temperature: float = 0.07,
    weights: Optional[torch.Tensor] = None,  # (N,) optional confidence weights in [0,1]
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Pairwise InfoNCE: match each student row with its teacher row.
    Negatives are other rows in the same batch (full N×N).
    """
    assert z_student.shape == z_teacher.shape
    z_s = F.normalize(z_student, dim=-1) if normalize else z_student
    z_t = F.normalize(z_teacher, dim=-1) if normalize else z_teacher

    logits = (z_s @ z_t.t()) / temperature  # (N,N)
    labels = torch.arange(z_s.shape[0], device=z_s.device, dtype=torch.long)

    if weights is None:
        loss = F.cross_entropy(logits, labels)
    else:
        # Weighted CE: average per-sample CE by weights
        ce = F.cross_entropy(logits, labels, reduction="none")  # (N,)
        w = weights.to(ce.dtype)
        loss = (w * ce).sum() / torch.clamp(w.sum(), min=1e-6)

    # diagnostics
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
    return {"loss": loss, "acc": acc, "logits": logits}
