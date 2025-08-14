from typing import Any, Dict, List

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from src.models.geometry_backbone.vggt_wrapper import VGGTWrapper
from src.models.geometry_backbone.geo_lifter import GeoLifter
from src.models.projectors import build_projectors
from src.models.semantic_backbone.clip_teacher import CLIPImageTeacher
from src.loss.info_nce import info_nce_pairwise


class PoCTrainer:
    """
    Minimal trainer for Tier-0 PoC:
      images (B=1,V,H,W,3) -> VGGTWrapper -> GeoLifter (N,2048) -> GeoProjector (N,clip_dim)
                              CLIPImageTeacher (B,V) -> per-point teacher (N,clip_dim)
                              -> InfoNCE
    We keep B=1 for PoC simplicity; extend batching as needed later.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Modules ---
        self.vggt = VGGTWrapper(cfg["vggt"]).to(self.device).eval()  # frozen
        self.lifter = GeoLifter(cfg["lift_geo"])                      # no params (no_grad inside)
        self.teacher = CLIPImageTeacher(cfg["clip_teacher"])          # frozen CLIP

        # Projector to CLIP space
        clip_dim = int(cfg["clip"]["embed_dim"])
        self.projectors = build_projectors(cfg, clip_embed_dim=clip_dim)
        self.projector_geo = self.projectors["geo"].to(self.device).train()

        # --- Optimizer ---
        opt_cfg = cfg["optim"]
        params = list(self.projector_geo.parameters())
        self.optimizer = optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])

        self.temperature = float(cfg["loss"]["temperature"])

        # torch AMP
        self.amp_dtype = torch.bfloat16 if cfg["general"].get("amp_dtype", "bf16") == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # --- TensorBoard (optional) ---
        log_cfg = cfg.get("logging", {})
        self.use_tb = bool(log_cfg.get("use_tensorboard", False))
        self.scalar_interval = int(log_cfg.get("scalar_interval", 10))
        self.image_interval = int(log_cfg.get("image_interval", 200))
        self.flush_interval = int(log_cfg.get("flush_interval", 100))
        self.max_images = int(log_cfg.get("max_images", 4))
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=log_cfg.get("log_dir", "runs/poc")) if self.use_tb else None

    def _images_to_pil_grid(self, images_bv: torch.Tensor) -> List[List["PIL.Image.Image"]]:
        """
        Convert a (B,V,3,H,W) uint8 tensor to nested list of PIL.Image.
        Assumes B=1 in PoC. If your dataloader already returns PILs, you can bypass this.
        """
        from torchvision.transforms.functional import to_pil_image
        B, V = images_bv.shape[:2]
        assert B == 1, "PoC assumes batch=1"
        out: List[List["PIL.Image.Image"]] = []
        for b in range(B):
            row = []
            for v in range(V):
                row.append(to_pil_image(images_bv[b, v]))
            out.append(row)
        return out

    @torch.no_grad()
    def _tb_log_images(self, images_bv: torch.Tensor):
        if not self.use_tb:
            return
        # images_bv: (1,V,3,H,W), dtype uint8 or float
        V = images_bv.shape[1]
        v_show = min(V, self.max_images)
        imgs = images_bv[0, :v_show]  # (v_show,3,H,W)
        if imgs.dtype == torch.uint8:
            imgs = imgs.to(torch.float32) / 255.0
        grid = make_grid(imgs, nrow=v_show, padding=2, normalize=False)
        self.writer.add_image("input/images_grid", grid, global_step=self.global_step)

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        batch should include:
          - images_bv: (B=1,V,3,H,W) uint8 or float [0..1]
          - paths (optional) for logging
        """
        images_bv = batch["images_bv"].to(self.device)
        B, V, C, H, W = images_bv.shape
        assert B == 1, "PoC assumes B=1"

        # --- VGGT forward (frozen) ---
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            vggt_out = self.vggt.forward(images_bv)  # dict with xyz_map, conf_map, patch_tokens_2048, meta, camera

        # --- Geo lift & sampling ---
        geo_pack = self.lifter.forward(vggt_out, scene_meta=None, batch_idx=0)
        feats_geo_raw = geo_pack["feats_geo_raw"].to(self.device)   # (N,2048), already L2-norm in default cfg
        weights = geo_pack.get("weights", None)                     # (N,) in [0,1]
        uv = geo_pack.get("uv", None)                               # (N,2) pixel (u,v) at VGGT size
        view_ids = geo_pack.get("view_ids", None)                   # (N,)
        H_vggt = vggt_out["meta"]["H"]
        W_vggt = vggt_out["meta"]["W"]

        # --- CLIP teacher (frozen) ---
        pil_bv = self._images_to_pil_grid(images_bv)
        teacher_out = self.teacher.forward(
            images_bv=pil_bv, uv=uv, view_ids=view_ids, vggt_hw=(H_vggt, W_vggt)
        )
        z_img_teacher = teacher_out["teacher_points"].to(self.device)  # (N, clip_dim)

        # --- Project student to CLIP space ---
        self.projector_geo.train()
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            z_geo = self.projector_geo(feats_geo_raw)  # (N, clip_dim)

            # InfoNCE (pairwise)
            loss_pack = info_nce_pairwise(
                z_student=z_geo,
                z_teacher=z_img_teacher,
                temperature=self.temperature,
                weights=weights
            )
            loss = loss_pack["loss"]

        # --- Backward ---
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # --- TensorBoard scalars ---
        if self.use_tb and (self.global_step % self.scalar_interval == 0):
            self.writer.add_scalar("train/loss", float(loss.detach().item()), self.global_step)
            self.writer.add_scalar("train/acc", float(loss_pack["acc"].detach().item()), self.global_step)
            self.writer.add_scalar("train/N_points", int(feats_geo_raw.shape[0]), self.global_step)
            # LR
            for i, g in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"lr/group_{i}", float(g["lr"]), self.global_step)

        # --- TensorBoard images ---
        if self.use_tb and (self.global_step % self.image_interval == 0):
            self._tb_log_images(images_bv)

        if self.use_tb and (self.global_step % self.flush_interval == 0):
            self.writer.flush()

        self.global_step += 1
        return {
            "loss": loss.detach().item(),
            "acc": loss_pack["acc"].detach().item(),
            "N_points": feats_geo_raw.shape[0],
        }

    def fit(self, loader, epochs: int, log_interval: int = 20):
        it = 0
        for ep in range(epochs):
            for batch in loader:
                out = self.training_step(batch)
                if it % log_interval == 0:
                    print(f"[ep {ep} it {it}] loss={out['loss']:.4f} acc={out['acc']:.3f} N={out['N_points']}")
                it += 1
        # --- NEW: close TensorBoard ---
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
