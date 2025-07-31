# Notes:
#   * LayerNorm is preferred by default for stability with variable point counts.
#   * L2 normalization at the end stabilizes cosine-based losses and inference.

# Why Linear -> Gelu:
#   * Use Linear -> Gelu -> Linear (double MLP) is potentially risky in overfitting.
#   * After PoC we might try ablations on double MLP + residuals

# PoC Ablation Study:
#   - "before PTv3": in_dim=2048 (VGGT-lifted per-point features), out_dim=clip_dim
#   - "after  PTv3": in_dim=C_ptv3 (PointTransformer output dim), out_dim=clip_dim

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, num_features: int) -> nn.Module:
    """
    Factory for normalization layers.
    Args:
        norm: "none" | "layernorm" | "batchnorm"
        num_features: number of channels to normalize
    Returns:
        nn.Module that is a normalization layer or nn.Identity().
    """
    norm = (norm or "none").lower()
    if norm == "layernorm":
        return nn.LayerNorm(num_features)
    if norm == "batchnorm":
        # BN expects (N, C) shape. Use with care for small N.
        return nn.BatchNorm1d(num_features)
    return nn.Identity()


def _make_activation(activation: str) -> nn.Module:
    """
    Factory for activation layers.
    Args:
        activation: "none" | "gelu"
    """
    activation = (activation or "none").lower()
    if activation == "gelu":
        return nn.GELU()
    return nn.Identity()


class GeoProjectorToCLIP(nn.Module):
    """
    Trainable projector that maps geometric point features into the native CLIP/SigLIP embedding space.

    Forward I/O:
        x: Tensor of shape (N, in_dim)
        returns: Tensor of shape (N, out_dim), optionally L2-normalized per point

    Recommended defaults for PoC:
        norm="layernorm", activation="none", dropout=0.0, l2norm=True.

    Why LayerNorm over BatchNorm?
        Point clouds often have variable sizes (N). LayerNorm is more robust in such settings
        and with small effective batches.

    Why L2-normalize at the end?
        Cosine similarity and InfoNCE in CLIP space assume unit-length embeddings.
        Normalizing at the output simplifies loss design and inference.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 norm: str = "layernorm",     # "none" | "layernorm" | "batchnorm"
                 activation: str = "none",    # "none" | "gelu"
                 dropout: float = 0.0,
                 l2norm: bool = True):
        super().__init__()
        assert in_dim > 0 and out_dim > 0, "in_dim and out_dim must be positive"

        self.out_dim = int(out_dim)
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = _make_norm(norm, out_dim)
        self.act = _make_activation(activation)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()
        self.l2norm = bool(l2norm)

        # Weight initialization: xavier uniform is a solid default for linear projections.
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim)
        Returns:
            z: (N, out_dim) (unit-length per row if l2norm=True)
        """
        if x.dim() != 2:
            raise ValueError(f"GeoProjectorToCLIP expects 2D input (N, C), got shape={tuple(x.shape)}")

        z = self.fc(x)          # (N, out_dim)
        z = self.norm(z)            # (N, out_dim)
        z = self.act(z)             # (N, out_dim)
        z = self.drop(z)            # (N, out_dim)
        if self.l2norm:
            # Epsilon is handled internally by F.normalize.
            z = F.normalize(z, dim=-1)
        return z


def build_projectors(cfg: Dict[str, Any],
                     clip_embed_dim: Optional[int] = None) -> Dict[str, nn.Module]:
    """
    Factory that creates the geometry projector with config defaults and (optionally) ties
    out_dim to the CLIP embedding dimension.

    Args:
        cfg: dict-like YAML with keys under cfg['model']['projector'].
             Expected fields (with PoC defaults):
               model.projector.geo_in_dim:   int, e.g., 2048
               model.projector.geo_out_dim:  int, e.g., 768 (will be overridden by clip_embed_dim if provided)
               model.projector.geo_norm:     "none"|"layernorm"|"batchnorm"
               model.projector.geo_activation:"none"|"gelu"
               model.projector.geo_dropout:  float
               model.projector.geo_l2norm:   bool
        clip_embed_dim: if provided, overrides geo_out_dim to this value. Set to CLIP/SigLIP embed dim.

    Returns:
        A dict of modules:
            {"geo": GeoProjectorToCLIP(...)}
    """
    mcfg = cfg.get("model", {}).get("projector", {})

    in_dim = int(mcfg.get("geo_in_dim", 2048))
    if clip_embed_dim is not None:
        out_dim = int(clip_embed_dim)
    else:
        out_dim = int(mcfg.get("geo_out_dim", 512))

    norm = str(mcfg.get("geo_norm", "layernorm"))
    activation = str(mcfg.get("geo_activation", "none"))
    dropout = float(mcfg.get("geo_dropout", 0.0))
    l2norm = bool(mcfg.get("geo_l2norm", True))

    proj = GeoProjectorToCLIP(
        in_dim=in_dim,
        out_dim=out_dim,
        norm=norm,
        activation=activation,
        dropout=dropout,
        l2norm=l2norm,
    )
    return {"geo": proj}
