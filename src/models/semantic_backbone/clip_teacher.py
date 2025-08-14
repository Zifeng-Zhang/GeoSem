import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
from transformers import CLIPModel, CLIPProcessor

from src.utils.feature import bilinear_from_tokens_hw


class CLIPImageTeacher(nn.Module):
    """
    Frozen CLIP image-side teacher that provides:
      1) per-view patch tokens in CLIP space (after 'visual_projection', dim=clip_dim)
      2) lifted per-point teacher features by rescaling UV from VGGT image space to CLIP image space
         and bilinearly sampling the CLIP patch grid.

    Typical usage:
        teacher = CLIPImageTeacher(cfg['clip_teacher'])
        out = teacher(images_bv, uv, view_ids, vggt_hw)  # -> per-point teacher (N, clip_dim)

    Args (cfg):
        model_name: HuggingFace id, e.g., "openai/clip-vit-large-patch14"
        image_size: int, CLIP input size (default 224 for ViT-L/14)
        patch_size: int, 14 for ViT-L/14 (do not change)
        normalize_out: bool, L2-normalize patch tokens and per-point teacher
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.model_name = cfg.get("model_name", "openai/clip-vit-large-patch14")
        self.image_size = int(cfg.get("image_size", 224))
        self.patch_size = int(cfg.get("patch_size", 14))
        self.normalize_out = bool(cfg.get("normalize_out", True))

        device_setting = cfg.get("device", "auto")
        if device_setting == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_setting)

        dtype_str = str(cfg.get("dtype", "fp16")).lower()
        if dtype_str == "bf16":
            self.autocast_dtype = torch.bfloat16
        elif dtype_str == "fp16" or dtype_str == "half":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

        # load CLIP
        self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model = self.model.eval().to(self.device)

        # pull dims
        self.clip_dim = self.model.config.projection_dim  # 768 for ViT-L/14
        self.hidden_size = self.model.vision_model.config.hidden_size  # 1024 for ViT-L/14

        # sanity
        assert self.patch_size == 14, "This teacher assumes ViT-L/14. For other models, adjust patch_size."

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_images_to_patch_tokens(
        self,
        images_bv: List[List["PIL.Image.Image"]],
    ) -> Dict[str, Any]:
        """
        Convert a batch of V-view PIL images to CLIP patch tokens in CLIP space.

        Args:
            images_bv: nested list length B of lists length V, each element a PIL.Image
        Returns:
            {
              "patch_tokens": (B, V, Hc, Wc, clip_dim),   # CLIP-embedded patch tokens
              "Hc": int, "Wc": int,
            }
        """
        B = len(images_bv)
        assert B > 0 and len(images_bv[0]) > 0, "Expect non-empty [B][V] images"
        V = len(images_bv[0])

        # Flatten for processor
        flat_images: List["PIL.Image.Image"] = []
        for b in range(B):
            assert len(images_bv[b]) == V
            for v in range(V):
                flat_images.append(images_bv[b][v])

        # Force CLIP-native square size to avoid pos-embed interpolation complexity
        inputs = self.processor(
            images=flat_images,
            return_tensors="pt",
            do_resize=True,
            size={"shortest_edge": self.image_size},  # for CLIPProcessor; makes square for PIL images
            do_center_crop=True
        )
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            # vision model forward (get per-token hidden states)
            vis_out = self.model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
            x = vis_out.last_hidden_state  # (B*V, 1+Npatch, hidden=1024)
            # drop CLS -> (B*V, Npatch, hidden)
            x = x[:, 1:, :]
            # apply CLIP visual projection to per-patch tokens -> CLIP space
            proj = self.model.visual_projection  # Linear(1024->clip_dim)
            x = proj(x)  # (B*V, Npatch, clip_dim)
            if self.normalize_out:
                x = F.normalize(x, dim=-1)

        # reshape to (B,V,Hc,Wc,clip_dim)
        Hc = self.image_size // self.patch_size
        Wc = self.image_size // self.patch_size
        x = x.view(B, V, Hc, Wc, self.clip_dim).contiguous()
        return {"patch_tokens": x, "Hc": Hc, "Wc": Wc}

    @torch.no_grad()
    def lift_to_points_bilinear(
        self,
        patch_tokens_bv: torch.Tensor,  # (B,V,Hc,Wc,clip_dim)
        uv: torch.Tensor,               # (N,2) UV in VGGT image space (u in [0..Wv-1], v in [0..Hv-1])
        view_ids: torch.Tensor,         # (N,) int in [0..V-1], which view each point came from
        vggt_hw: Tuple[int, int],       # (Hv, Wv) corresponding to VGGT processed size
    ) -> torch.Tensor:
        """
        Lift CLIP patch tokens to per-point teacher features, by:
         1) selecting the corresponding view v for each point
         2) rescaling UV from VGGT (Hv,Wv) to CLIP image size (image_size,image_size)
         3) mapping pixel coords to patch-grid coords and bilinear sampling.

        Returns:
            feats_teacher: (N, clip_dim)
        """
        device = patch_tokens_bv.device
        B, V, Hc, Wc, C = patch_tokens_bv.shape
        Hv, Wv = vggt_hw
        assert uv.shape[0] == view_ids.shape[0], "uv and view_ids must have same N"

        N = uv.shape[0]
        # rescale (u,v) from VGGT to CLIP pixel space
        scale_u = float(self.image_size) / float(Wv)
        scale_v = float(self.image_size) / float(Hv)
        u_clip = uv[:, 0].to(torch.float32) * scale_u
        v_clip = uv[:, 1].to(torch.float32) * scale_v

        # map pixel to patch continuous coords: floor(pixel/patch_size)
        xp = u_clip / float(self.patch_size)  # in [0..Wc)
        yp = v_clip / float(self.patch_size)  # in [0..Hc)

        # gather tokens by view id
        # we assume a single global batch (B==1) in PoC (extend as needed)
        assert B == 1, "PoC assumes B=1. Extend indexing for multi-B if needed."
        feats = torch.empty((N, C), device=device, dtype=patch_tokens_bv.dtype)
        for v in range(V):
            mask = (view_ids == v)
            if mask.any():
                tokens_hw_c = patch_tokens_bv[0, v]  # (Hc,Wc,C)
                feats[mask] = bilinear_from_tokens_hw(
                    tokens_hw_c,
                    xp[mask].to(tokens_hw_c.dtype).to(device),
                    yp[mask].to(tokens_hw_c.dtype).to(device),
                    align_corners=False,
                    clamp=True
                )
        if self.normalize_out:
            feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def forward(
        self,
        images_bv: List[List["PIL.Image.Image"]],
        uv: torch.Tensor,
        view_ids: torch.Tensor,
        vggt_hw: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Convenience wrapper: image -> patch_tokens -> lifted per-point teacher.
        Returns:
          {
            "patch_tokens": (B,V,Hc,Wc,clip_dim),
            "teacher_points": (N,clip_dim)
          }
        """
        pack = self.encode_images_to_patch_tokens(images_bv)
        teacher_pts = self.lift_to_points_bilinear(
            pack["patch_tokens"], uv=uv, view_ids=view_ids, vggt_hw=vggt_hw
        )
        return {"patch_tokens": pack["patch_tokens"], "teacher_points": teacher_pts}
