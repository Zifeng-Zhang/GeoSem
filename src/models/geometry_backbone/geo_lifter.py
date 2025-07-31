import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from src.utils.geometry import inverse_extrinsics, farthest_point_sampling


def _apply_conf_activation(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Map raw confidence to a usable weight/probability.
    mode: "sigmoid" | "softplus" | "none"
    """
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "softplus":
        return F.softplus(x)
    return x  # "none"


class GeoLifter:
    """
    Lift VGGT per-patch tokens to per-point tokens using (xyz_map, conf_map).
    The module performs:
      1) confidence-based pixel sampling to select ~N points
      2) bi-linear interpolation on (H_p, W_p, C=2048) patch tokens to get per-point features
      3) (optional) camera->world transform for coordinates
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        cfg expects the 'lift_geo' section of the YAML (see default.yaml).
        """
        self.stride: int = int(cfg.get("stride", 14))
        self.use_world_coords: bool = bool(cfg.get("use_world_coords", False))

        # confidence handling
        self.conf_activation: str = str(cfg.get("conf_activation", "sigmoid"))
        self.conf_threshold: float = float(cfg.get("conf_threshold", 0.5))

        # sampling
        self.max_points: int = int(cfg.get("max_points", 16000))
        self.per_view_quota: int = int(cfg.get("per_view_quota", -1))  # -1 means auto
        self.strategy: str = str(cfg.get("strategy", "topk_conf"))  # "topk_conf"|"uniform"|"fps"
        self.fps_after_topk: bool = bool(cfg.get("fps_after_topk", False))
        self.random_seed: int = int(cfg.get("random_seed", 0))

        # bilinear sampling options
        self.align_corners: bool = bool(cfg.get("align_corners", False))
        self.clamp_grid: bool = bool(cfg.get("clamp_grid", True))

        # dedup(de-duplication) (disabled by default in PoC)
        self.dedup_cfg = cfg.get("dedup", {"enabled": False})

        # feature post-processing
        self.norm_feat: str = str(cfg.get("normalize_feature", "l2"))
        self.return_weights: bool = bool(cfg.get("return_weights", True))
        self.return_uv: bool = bool(cfg.get("return_uv", True))
        self.return_view_ids: bool = bool(cfg.get("return_view_ids", True))

    @torch.no_grad()
    def forward(
        self,
        vggt_out: Dict[str, Any],
        scene_meta: Optional[Dict[str, Any]] = None,
        batch_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        vggt_out must contain:
            xyz_map (B,V,H,W,3)
            conf_map (B,V,H,W)
            patch_tokens_2048 (B,V,H_p,W_p,2048)
            meta {H,W,H_p,W_p,patch_stride}
            camera {extrinsic (B,V,4,4) or None}
        Returns a dict with coords, feats_geo_raw, and bookkeeping.
        """
        # unpack inputs
        xyz_map: torch.Tensor = vggt_out["xyz_map"]       # (B,V,H,W,3)
        conf_map: torch.Tensor = vggt_out["conf_map"]     # (B,V,H,W)
        patch_tokens: torch.Tensor = vggt_out["patch_tokens_2048"]  # (B,V,H_p,W_p,2048)
        meta: Dict[str, Any] = vggt_out.get("meta", {})
        camera: Dict[str, Any] = vggt_out.get("camera", {})

        B, V, H, W, _ = xyz_map.shape
        _, _, H_p, W_p, C = patch_tokens.shape
        assert C == 2048, f"Expected token dim=2048, got={C}"
        stride = int(meta.get("patch_stride", self.stride))
        assert stride == self.stride, "Stride mismatch between wrapper and lifter"

        device = xyz_map.device
        dtype = xyz_map.dtype

        # apply confidence activation if needed (only for sampling & weights)
        conf_act = _apply_conf_activation(conf_map, self.conf_activation)

        # compute total valid pixels across views (thresholded after activation)
        valid_mask = conf_act >= self.conf_threshold  # (B,V,H,W)
        valid_counts_per_view = valid_mask.view(B, V, -1).sum(dim=-1)  # (B,V)

        # per-view quotas
        quotas = self._compute_view_quotas(valid_counts_per_view, self.max_points)  # (B,V)

        # will collect outputs across (B,V)
        all_coords: List[torch.Tensor] = []
        all_feats: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []
        all_uv: List[torch.Tensor] = []
        all_vid: List[torch.Tensor] = []
        all_bidx: List[torch.Tensor] = []

        # precompute Twc if needed
        Twc = None
        if self.use_world_coords and camera is not None and camera.get("extrinsic", None) is not None:
            Tcw = camera["extrinsic"]  # (B,V,4,4)
            Twc = inverse_extrinsics(Tcw)  # (B,V,4,4)

        # iterate over batch and views
        for b in range(B):
            for v in range(V):
                k_quota = int(quotas[b, v].item())
                if k_quota <= 0:
                    continue

                conf_view = conf_act[b, v]           # (H,W)
                valid_view = valid_mask[b, v]        # (H,W)
                if valid_view.sum() == 0:
                    continue

                # 1) get candidate indices
                flat_conf = conf_view[valid_view]  # (Nv,)
                # (u,v) coords for valid positions
                vv, uu = torch.nonzero(valid_view, as_tuple=True)  # (Nv,)
                # NOTE: our uv convention is (u,v) where u in [0..W-1], v in [0..H-1]
                # uu: x (width), vv: y (height)

                # 2) select by strategy
                sel_idx = self._select_indices(flat_conf, uu, vv, k_quota, xyz_map[b, v], strategy=self.strategy)

                # selected pixel coords
                uu_sel = uu[sel_idx]
                vv_sel = vv[sel_idx]
                # selected confidence weights (post activation)
                w_sel = flat_conf[sel_idx]  # (k_quota,)

                # 3) fetch xyz (camera coords)
                xyz_sel = xyz_map[b, v, vv_sel, uu_sel, :]  # (k_quota,3)

                # (optional) transform to world coords
                if self.use_world_coords and Twc is not None:
                    # homogeneous transform X_w = Twc @ [X_c;1]
                    ones = torch.ones((xyz_sel.shape[0], 1), device=device, dtype=xyz_sel.dtype)
                    Xc_h = torch.cat([xyz_sel, ones], dim=1)  # (N,4)
                    Xw_h = (Twc[b, v] @ Xc_h.T).T            # (N,4)
                    xyz_sel = Xw_h[:, :3] / torch.clamp(Xw_h[:, 3:4], min=1e-8)

                # 4) bilinear sample per-patch tokens -> per-point features
                #    tokens_view: (H_p, W_p, 2048)
                feats_sel = self._bilinear_from_tokens(
                    patch_tokens[b, v], uu_sel, vv_sel, H, W, H_p, W_p, stride,
                    align_corners=self.align_corners, clamp_grid=self.clamp_grid
                )  # (k_quota, 2048)

                # 5) (optional) feature normalization
                if self.norm_feat == "l2":
                    feats_sel = F.normalize(feats_sel, dim=-1)

                # bookkeeping
                all_coords.append(xyz_sel)                 # (k,3)
                all_feats.append(feats_sel)               # (k,2048)
                if self.return_weights:
                    all_weights.append(w_sel)             # (k,)
                if self.return_uv:
                    all_uv.append(torch.stack([uu_sel, vv_sel], dim=1).to(torch.long))  # (k,2)
                if self.return_view_ids:
                    all_vid.append(torch.full((k_quota,), v, device=device, dtype=torch.long))
                if batch_idx is not None:
                    all_bidx.append(torch.full((k_quota,), int(batch_idx), device=device, dtype=torch.long))

        # concatenate
        coords = torch.cat(all_coords, dim=0) if all_coords else torch.empty(0, 3, device=device, dtype=dtype)
        feats = torch.cat(all_feats, dim=0) if all_feats else torch.empty(0, 2048, device=device, dtype=dtype)
        weights = torch.cat(all_weights, dim=0) if (all_weights and self.return_weights) else None
        uv = torch.cat(all_uv, dim=0) if (all_uv and self.return_uv) else None
        view_ids = torch.cat(all_vid, dim=0) if (all_vid and self.return_view_ids) else None
        bidx = torch.cat(all_bidx, dim=0) if all_bidx else None

        out: Dict[str, Any] = {
            "coords": coords,              # (N,3) camera/world
            "feats_geo_raw": feats,        # (N,2048)
            "weights": weights,            # (N,) or None
            "uv": uv,                      # (N,2) or None
            "view_ids": view_ids,          # (N,) or None
            "batch_idx": bidx,             # (N,) or None
            "meta": {
                "H": H, "W": W,
                "H_p": H_p, "W_p": W_p,
                "stride": stride,
                "coord_frame": "world" if self.use_world_coords else "camera",
                "sampling": {
                    "strategy": self.strategy,
                    "max_points": self.max_points,
                    "per_view_quota": self.per_view_quota,
                    "conf_activation": self.conf_activation,
                    "conf_threshold": self.conf_threshold,
                    "fps_after_topk": self.fps_after_topk,
                }
            },
            "camera": {
                "extrinsic": camera.get("extrinsic", None),
                "intrinsic": camera.get("intrinsic", None),
            }
        }
        return out

    # ------------------------- helpers -------------------------

    def _compute_view_quotas(self, valid_counts: torch.Tensor, max_points: int) -> torch.Tensor:
        """
        Distribute max_points across (B,V) according to valid pixel counts.
        valid_counts: (B,V)
        returns quotas: (B,V) (sum per batch <= max_points; last view absorbs remainder)
        """
        B, V = valid_counts.shape
        quotas = torch.zeros_like(valid_counts, dtype=torch.long)
        for b in range(B):
            total = int(valid_counts[b].sum().item())
            if total <= 0 or max_points <= 0:
                continue
            if self.per_view_quota > 0:
                # fixed per-view cap
                quotas[b] = torch.clamp(valid_counts[b], max=self.per_view_quota).to(torch.long)
            else:
                # proportional allocation
                remain = max_points
                # avoid division by zero
                weights = valid_counts[b].float() / max(total, 1)
                alloc = torch.floor(weights * max_points).to(torch.long)
                # ensure at least 1 for non-zero views
                nonzero_mask = valid_counts[b] > 0
                alloc[nonzero_mask] = torch.clamp(alloc[nonzero_mask], min=1)
                # adjust remainder
                s = int(alloc.sum().item())
                if s > max_points:
                    # shrink the largest allocations
                    overflow = s - max_points
                    # reduce one-by-one from views with largest alloc
                    vals, idxs = torch.sort(alloc, descending=True)
                    for i in range(V):
                        if overflow <= 0:
                            break
                        take = min(overflow, int(vals[i].item()))
                        alloc[idxs[i]] -= take
                        overflow -= take
                elif s < max_points:
                    # distribute remainder to the views with most valids
                    remainder = max_points - s
                    _, order = torch.sort(valid_counts[b], descending=True)
                    i = 0
                    while remainder > 0 and i < V:
                        alloc[order[i]] += 1
                        remainder -= 1
                        i = (i + 1) % V
                quotas[b] = alloc
        return quotas

    def _select_indices(
        self,
        flat_conf: torch.Tensor,
        uu: torch.Tensor,
        vv: torch.Tensor,
        k_quota: int,
        xyz_view: torch.Tensor,
        strategy: str = "topk_conf",
    ) -> torch.Tensor:
        """
        Select k_quota indices from candidates (flat_conf, uu, vv).
        strategy: "topk_conf" | "uniform" | "fps"
        xyz_view: (H,W,3) used by FPS (in 3D space).
        Returns indices into (uu, vv, flat_conf).
        """
        N = flat_conf.numel()
        k_quota = min(k_quota, N)
        if k_quota <= 0:
            return torch.empty(0, dtype=torch.long, device=flat_conf.device)

        if strategy == "uniform":
            # uniform random over valid pixels
            g = torch.Generator(device=flat_conf.device)
            g.manual_seed(self.random_seed)
            perm = torch.randperm(N, generator=g, device=flat_conf.device)
            return perm[:k_quota]

        # default: topk_conf (+ optional FPS)
        vals, idxs = torch.topk(flat_conf, k_quota if not self.fps_after_topk else min(2 * k_quota, N))
        if self.fps_after_topk:
            # run FPS on the picked candidates to reduce to k_quota
            uu_k = uu[idxs]
            vv_k = vv[idxs]
            xyz_k = xyz_view[vv_k, uu_k, :]  # (K,3)
            fps_idx = farthest_point_sampling(xyz_k, k_quota, seed=self.random_seed)  # indices into K
            return idxs[fps_idx]
        return idxs

    def _bilinear_from_tokens(
        self,
        tokens_hw_c: torch.Tensor,  # (H_p, W_p, C)
        uu: torch.Tensor,           # (N,) pixel x in [0..W-1]
        vv: torch.Tensor,           # (N,) pixel y in [0..H-1]
        H: int, W: int, H_p: int, W_p: int, stride: int,
        align_corners: bool, clamp_grid: bool
    ) -> torch.Tensor:
        """
        Vectorized bilinear interpolation from patch grid to arbitrary pixel positions.
        We avoid grid_sample loops by computing neighbors explicitly.
        tokens_hw_c: (H_p, W_p, C)
        Returns feats: (N, C)
        """
        device = tokens_hw_c.device
        dtype = tokens_hw_c.dtype
        C = tokens_hw_c.shape[-1]

        # map pixel (u,v) to continuous patch grid (x_p,y_p)
        if align_corners:
            # interpret token centers at integer grid locations,
            # map pixel centers to grid centers (approximation):
            # x_p = (u + 0.5)/stride - 0.5; y_p similar
            xp = (uu.to(dtype) + 0.5) / float(stride) - 0.5
            yp = (vv.to(dtype) + 0.5) / float(stride) - 0.5
        else:
            # simple proportional mapping
            xp = uu.to(dtype) / float(stride)
            yp = vv.to(dtype) / float(stride)

        # clamp so that ix+1/iy+1 exist
        if clamp_grid:
            eps = 1e-6
            xp = torch.clamp(xp, 0.0, float(W_p - 1) - eps)
            yp = torch.clamp(yp, 0.0, float(H_p - 1) - eps)

        ix0 = torch.floor(xp).to(torch.long)              # (N,)
        iy0 = torch.floor(yp).to(torch.long)              # (N,)
        ix1 = torch.clamp(ix0 + 1, max=W_p - 1)
        iy1 = torch.clamp(iy0 + 1, max=H_p - 1)

        tx = (xp - ix0.to(dtype)).unsqueeze(1)            # (N,1)
        ty = (yp - iy0.to(dtype)).unsqueeze(1)            # (N,1)

        # gather 4 neighbors
        # tokens: (H_p, W_p, C) -> flatten indices
        # idx = y * W_p + x
        idx00 = iy0 * W_p + ix0                            # (N,)
        idx10 = iy0 * W_p + ix1
        idx01 = iy1 * W_p + ix0
        idx11 = iy1 * W_p + ix1

        flat = tokens_hw_c.view(H_p * W_p, C)              # (H_p*W_p, C)
        f00 = flat[idx00]                                  # (N,C)
        f10 = flat[idx10]
        f01 = flat[idx01]
        f11 = flat[idx11]

        # bilinear weights
        w00 = (1.0 - tx) * (1.0 - ty)                      # (N,1)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        print((w00 + w10 + w01 + w11 - 1).abs().max().item())

        feats = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11  # (N,C)
        return feats
