import torch
import math
import logging
from typing import Any, Dict, List, Optional

from third_party.vggt.vggt.models.vggt import VGGT
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from third_party.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from third_party.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map


logger = logging.getLogger(__name__)


def _pick_dtype(prefer_bf16: bool) -> torch.dtype:
    """
    Choose bf16 on Ampere+ when available (if prefer_bf16=True), otherwise fp16.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    if prefer_bf16 and major >= 8:
        return torch.bfloat16
    return torch.float16


class VGGTWrapper:
    """
    Thin wrapper around VGGT that:
      1) loads a pretrained model
      2) runs inference with autocast
      3) returns a normalized dict of tensors needed by downstream modules

    Public usage:
      vggt = VGGTWrapper(cfg['model']['vggt'], device=...); vggt.eval()
      out = vggt.forward_from_paths(["img1.png","img2.png"])
      # or if you already have a preprocessed tensor (V,3,H,W) or (B,V,3,H,W):
      out = vggt.forward(images)
    """

    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None):
        """
        Args:
          cfg: config dict under cfg['model']['vggt'] (see default.yaml)
          device: 'cuda' | 'cpu' | None (None -> infer from cfg/general)
        """
        self.ckpt_path: str = cfg.get("ckpt_path", "")
        self.eval_mode: bool = bool(cfg.get("eval_mode", True))
        self.prefer_bf16: bool = bool(cfg.get("prefer_bf16", True))
        self.patch_stride: int = int(cfg.get("patch_stride", 14))
        self.num_reg_tokens: int = int(cfg.get("num_reg_tokens", 5))
        self.return_depth_branch: bool = bool(cfg.get("return_depth_branch", True))
        self.return_camera_branch: bool = bool(cfg.get("return_camera_branch", True))
        self.return_unprojected_points: bool = bool(cfg.get("return_unprojected_points", False))

        # choose device
        if device is None:
            general = cfg.get("__general__", {})  # injected by caller if needed
            device_setting = general.get("device", "auto")
            if device_setting == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device_setting
        else:
            self.device = device

        # pick dtype for autocast
        self.amp_dtype: torch.dtype = _pick_dtype(self.prefer_bf16)

        # load model
        logger.info(f"[VGGTWrapper] Loading weights from: {self.ckpt_path}")
        self.model: VGGT = VGGT.from_pretrained(self.ckpt_path).to(self.device)

        if self.eval_mode:
            self.model.eval()

        # small sanity
        if not hasattr(self.model, "aggregator"):
            raise RuntimeError("VGGT model missing 'aggregator' module")

    # ------------------------------ public API ------------------------------

    @torch.no_grad()
    def forward_from_paths(
        self,
        image_paths: List[str],
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience entry: load + pre-process images (single scene window) then forward.
        Returns the same dict as `forward(...)`.
        """
        imgs = load_and_preprocess_images(image_paths, **(preprocess_kwargs or {})).to(self.device)
        # expected shape from loader: (V,3,H,W). Add batch dim.
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(0)
        return self.forward(imgs)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
          images: Tensor of shape (B,V,3,H,W) or (V,3,H,W). If (V,3,H,W), will add B=1.
          intrinsics: optional (B,V,3,3), if you want to override/inject dataset intrinsics.
                      (VGGT does not strictly require this for its heads, but you might store it.)

        Returns: a dict with keys:
          - xyz_map:        (B,V,H,W,3)
          - conf_map:       (B,V,H,W)
          - depth_map:      (B,V,H,W) or None
          - depth_conf:     (B,V,H,W) or None
          - camera:         dict with extrinsic (B,V,4,4), intrinsic (B,V,3,3), pose_enc (B,V,9)
          - patch_tokens_2048: (B,V,H_p,W_p,2048), reg tokens removed
          - meta:           dict with H,W,H_p,W_p,patch_stride,num_reg_tokens,ps_idx
          - xyz_from_unprojection: (B,V,H,W,3) if enabled, else None
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)  # -> (1,V,3,H,W)

        B, V, _, H, W = images.shape

        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=images.is_cuda):

            # 1) Aggregator (tokens for all heads)
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)
            last_tokens = aggregated_tokens_list[-1]  # (B,V,L,2048)
            if last_tokens.shape[-1] != 2048:
                raise RuntimeError(f"Expected token dim 2048, got {last_tokens.shape[-1]}")

            # 2) Camera head (optional)
            camera_dict = {"extrinsic": None, "intrinsic": None, "pose_enc": None}
            if self.return_camera_branch:
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]  # (B,V,9)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                camera_dict = {
                    "extrinsic": extrinsic,      # (B,V,4,4)
                    "intrinsic": intrinsic,      # (B,V,3,3)
                    "pose_enc": pose_enc,        # (B,V,9)
                }
            else:
                pose_enc = None

            # 3) Depth head (optional)
            if self.return_depth_branch:
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
                # ensure shapes (B,V,H,W)
                depth_map = depth_map.view(B, V, H, W)
                depth_conf = depth_conf.view(B, V, H, W)
            else:
                depth_map, depth_conf = None, None

            # 4) Point head (xyz + conf). This is what we primarily use.
            point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
            xyz_map = point_map.view(B, V, H, W, 3)     # (B,V,H,W,3)
            conf_map = point_conf.view(B, V, H, W)      # (B,V,H,W)

            # 5) (Optional) Recompute 3D via unprojecting depth + camera.
            # As suggested by the author in their paper and code repo,
            # projection with depth+camera poses slightly better than point_map
            if self.return_unprojected_points and (depth_map is not None) and (camera_dict["extrinsic"] is not None):
                # unproject_depth_map_to_point_map expects shapes without batch; run per (B,V)
                xyz_from_unproj = []
                for b in range(B):
                    xyz_from_unproj_v = []
                    for v in range(V):
                        xyz_uv = unproject_depth_map_to_point_map(
                            depth_map[b, v],         # (H,W)
                            camera_dict["extrinsic"][b, v],
                            camera_dict["intrinsic"][b, v],
                        )  # (H,W,3)
                        xyz_from_unproj_v.append(xyz_uv)
                    xyz_from_unproj.append(torch.stack(xyz_from_unproj_v, dim=0))
                xyz_from_unproj = torch.stack(xyz_from_unproj, dim=0)  # (B,V,H,W,3)
            else:
                xyz_from_unproj = None

            # 6) Patch tokens (remove reg tokens and reshape to grid)
            # last_tokens: (B,V,L,2048) where L = H_p*W_p + num_reg_tokens
            # discard 5 reg tokens for the vggt_wrapper output
            num_reg = self.num_reg_tokens
            seq_len = last_tokens.shape[-2]
            grid_len = seq_len - num_reg
            # derive H_p, W_p from (H,W) and stride if possible; fall back to sqrt when not divisible.
            Hp = H // self.patch_stride
            Wp = W // self.patch_stride
            if Hp * Wp != grid_len:
                # Fallback: try to infer a plausible grid from L
                root = int(math.sqrt(grid_len))
                if root * root != grid_len:
                    logger.warning(
                        f"[VGGTWrapper] Unexpected token length: L={seq_len}, "
                        f"grid_len={grid_len}. Can't factor into grid using stride={self.patch_stride}. "
                        f"Falling back to (Hp={root}, Wp={grid_len//root})."
                    )
                Hp = root
                Wp = grid_len // root

            patch_tokens = last_tokens[:, :, num_reg:, :]                      # (B,V,Hp*Wp,2048)
            patch_tokens = patch_tokens.view(B, V, Hp, Wp, 2048).contiguous()  # (B,V,Hp,Wp,2048)

        # Package outputs
        outputs: Dict[str, Any] = {
            "xyz_map": xyz_map,                     # (B,V,H,W,3)
            "conf_map": conf_map,                   # (B,V,H,W)
            "depth_map": depth_map,                 # (B,V,H,W) or None
            "depth_conf": depth_conf,               # (B,V,H,W) or None
            "camera": camera_dict,                  # dict of extrinsic/intrinsic/pose_enc
            "patch_tokens_2048": patch_tokens,      # (B,V,Hp,Wp,2048)
            "meta": {
                "H": H, "W": W,
                "H_p": Hp, "W_p": Wp,
                "patch_stride": self.patch_stride,
                "num_reg_tokens": self.num_reg_tokens,
                "ps_idx": ps_idx,                   # keep original for advanced heads if needed
            },
            "xyz_from_unprojection": xyz_from_unproj,  # (B,V,H,W,3) or None
        }

        return outputs

    # ------------------------------ Helper Functions ------------------------------
    def to(self, device: str, dtype: Optional[torch.dtype] = None) -> "VGGTWrapper":
        self.device = device
        self.model.to(device)
        if dtype is not None:
            self.amp_dtype = dtype
        return self

    def eval(self) -> "VGGTWrapper":
        self.model.eval()
        return self
