import os
import glob
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except Exception as e:
    Image = None


def _listdir_numeric_jpgs(color_dir: str) -> List[Tuple[int, str]]:
    """
    ScanNet++ export style:
      color/
        0.jpg, 20.jpg, 40.jpg, ...
    Return a list of (frame_id, full_path), sorted by frame_id (int).
    """
    paths = glob.glob(os.path.join(color_dir, "*.jpg"))
    out: List[Tuple[int, str]] = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            fid = int(name)
        except ValueError:
            # ignore non-numeric files
            continue
        out.append((fid, p))
    out.sort(key=lambda x: x[0])
    return out


def _load_image_as_tensor(path: str,
                          convert_to_rgb: bool = True,
                          return_uint8: bool = True,
                          keep_aspect: bool = True,
                          max_side: Optional[int] = None) -> torch.Tensor:
    """
    Load an image from path and return a tensor of shape (3, H, W).
    - return_uint8=True  -> dtype=torch.uint8, range [0,255]
    - return_uint8=False -> dtype=torch.float32, range [0,1]
    - keep_aspect=True with max_side: downscale if max(H,W) > max_side (maintain aspect ratio)
    """
    assert Image is not None, "PIL is required to load images"

    img = Image.open(path)
    if convert_to_rgb:
        img = img.convert("RGB")

    if max_side is not None:
        w, h = img.size
        max_hw = max(w, h)
        if max_hw > max_side:
            scale = max_side / float(max_hw)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Convert to tensor
    # PIL -> numpy -> torch
    import numpy as np
    arr = np.array(img)  # (H, W, 3), uint8
    if return_uint8:
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    else:
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
    return t


class ScanNetStreamDataset(Dataset):
    """
    Stream scenes from ScanNet++ (or v2 2D dump) and sample V frames per item.

    Supported sampling strategies:
      - sequential_window: sliding windows with (window_stride) and intra-window (temporal_stride)
      - random_window:     per scene sample epoch_len_per_scene random windows
      - random_pairs:      only valid when V == 2; sample epoch_len_per_scene pairs (i<j),
                           optionally constrained by max_temporal_gap

    Outputs:
      if io.output_mode == "tensor":
        {
          "images_bv": Tensor (1,V,3,H,W) or collate -> (B,V,3,H,W),
          "scene_id": str,
          "frame_ids": List[int],
          "image_paths": List[str],
          "orig_hw": List[Tuple[int,int]],
        }
      if io.output_mode == "paths":
        {
          "image_paths": List[str],
          "scene_id": str,
          "frame_ids": List[int],
          "orig_hw": List[Tuple[int,int]],
        }
    """

    def __init__(self, cfg: Dict[str, Any], split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        dcfg = cfg["data"]
        self.root: str = dcfg["root"]
        self.scenes_spec: Union[List[str], str] = dcfg["scene_ids"]
        self.frames_per_sample: int = int(dcfg.get("frames_per_sample", 2))

        scfg = dcfg.get("sampling", {})
        self.strategy: str = scfg.get("strategy", "sequential_window")
        self.window_stride: int = int(scfg.get("window_stride", 1))
        self.temporal_stride: int = int(scfg.get("temporal_stride", 1))
        self.epoch_len_per_scene: int = int(scfg.get("epoch_len_per_scene", 500))
        self.max_temporal_gap: Optional[int] = scfg.get("max_temporal_gap", None)
        if self.max_temporal_gap is not None:
            self.max_temporal_gap = int(self.max_temporal_gap)

        iocfg = dcfg.get("io", {})
        self.output_mode: str = iocfg.get("output_mode", "tensor")   # "tensor" | "paths"
        self.return_uint8: bool = bool(iocfg.get("return_uint8", True))
        self.convert_to_rgb: bool = bool(iocfg.get("convert_to_rgb", True))
        self.keep_aspect: bool = bool(iocfg.get("keep_aspect", True))
        self.max_side: Optional[int] = iocfg.get("max_side", None)
        if self.max_side is not None:
            self.max_side = int(self.max_side)

        hcfg = dcfg.get("dataloader_hints", {})
        self.shuffle_scenes: bool = bool(hcfg.get("shuffle_scenes", True))
        self.seed: int = int(hcfg.get("seed", 0))

        # Resolve scene list (allow txt file or inline list)
        if isinstance(self.scenes_spec, str) and os.path.isfile(self.scenes_spec):
            with open(self.scenes_spec, "r") as f:
                self.scene_ids = [line.strip() for line in f if line.strip()]
        elif isinstance(self.scenes_spec, list):
            self.scene_ids = list(self.scenes_spec)
        else:
            raise ValueError("data.scene_ids must be a list of scene IDs or a path to a .txt file")

        # Build per-scene frame tables
        self.scenes: List[Dict[str, Any]] = []
        for sid in self.scene_ids:
            color_dir = os.path.join(self.root, sid, "color")
            if not os.path.isdir(color_dir):
                raise FileNotFoundError(f"Missing color dir: {color_dir}")
            frames = _listdir_numeric_jpgs(color_dir)
            if len(frames) < self.frames_per_sample:
                # skip scenes with too few frames
                continue
            self.scenes.append({
                "scene_id": sid,
                "color_dir": color_dir,
                "frames": frames,  # List[(fid:int, path:str)]
            })

        if len(self.scenes) == 0:
            raise RuntimeError("No valid scenes found under root with enough frames")

        # RNG and precomputed index list
        self.rng = random.Random(self.seed)
        self.samples: List[Tuple[int, List[int]]] = []  # (scene_idx, list of local indices into frames[])
        self.refresh_epoch(epoch=0)

    # ----------------- epoch sampling control -----------------

    def refresh_epoch(self, epoch: int = 0):
        """
        Rebuild self.samples for the current epoch, enabling true randomness
        for random_* strategies while keeping determinism (seeded by epoch).
        Call this at the start of each epoch in your trainer.
        """
        self.rng.seed(self.seed + epoch)
        samples: List[Tuple[int, List[int]]] = []

        if self.strategy == "sequential_window":
            for s_idx, srec in enumerate(self.scenes):
                num = len(srec["frames"])
                V = self.frames_per_sample
                stride = self.temporal_stride
                # windows defined over index positions (not the numeric frame_id)
                max_start = num - (V - 1) * stride
                for i in range(0, max_start, self.window_stride):
                    idxs = [i + k * stride for k in range(V)]
                    samples.append((s_idx, idxs))

        elif self.strategy == "random_window":
            for s_idx, srec in enumerate(self.scenes):
                num = len(srec["frames"])
                V = self.frames_per_sample
                stride = self.temporal_stride
                need = self.epoch_len_per_scene
                count = 0
                while count < need:
                    # choose a start index i so that i + (V-1)*stride < num
                    max_start = num - (V - 1) * stride
                    if max_start <= 0:
                        break
                    i = self.rng.randrange(0, max_start)
                    idxs = [i + k * stride for k in range(V)]
                    samples.append((s_idx, idxs))
                    count += 1

        elif self.strategy == "random_pairs":
            assert self.frames_per_sample == 2, "random_pairs requires frames_per_sample == 2"
            for s_idx, srec in enumerate(self.scenes):
                num = len(srec["frames"])
                need = self.epoch_len_per_scene
                count = 0
                while count < need:
                    i = self.rng.randrange(0, num - 1)
                    if self.max_temporal_gap is None:
                        j = self.rng.randrange(i + 1, num)
                    else:
                        jmax = min(num, i + 1 + self.max_temporal_gap)
                        if jmax <= i + 1:
                            continue
                        j = self.rng.randrange(i + 1, jmax)
                    samples.append((s_idx, [i, j]))
                    count += 1
        else:
            raise ValueError(f"Unknown sampling.strategy: {self.strategy}")

        # Optionally shuffle scene order (acts on the flat sample list)
        if self.shuffle_scenes:
            self.rng.shuffle(samples)

        self.samples = samples

    # ----------------- dataset protocol -----------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames_tensor(self, paths: List[str]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Load paths -> Tensor (V,3,H,W); returns also list of (H,W) for each frame.
        """
        tensors: List[torch.Tensor] = []
        hw_list: List[Tuple[int, int]] = []
        for p in paths:
            t = _load_image_as_tensor(
                p,
                convert_to_rgb=self.convert_to_rgb,
                return_uint8=self.return_uint8,
                keep_aspect=self.keep_aspect,
                max_side=self.max_side,
            )
            tensors.append(t)
            _, H, W = t.shape
            hw_list.append((H, W))
        imgs_v = torch.stack(tensors, dim=0)  # (V,3,H,W)  (assumes same size; true for ScanNet++)
        return imgs_v, hw_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s_idx, local_idxs = self.samples[idx]
        srec = self.scenes[s_idx]
        scene_id = srec["scene_id"]
        frames = srec["frames"]

        # Resolve per-frame paths and numeric frame_ids
        sel = [frames[i] for i in local_idxs]
        frame_ids = [fid for (fid, _) in sel]
        paths = [p for (_, p) in sel]

        # Most ScanNet++ rgb are same size (320x240). We still record per-frame hw.
        if self.output_mode == "tensor":
            imgs_v, hw_list = self._load_frames_tensor(paths)  # (V,3,H,W)
            # Add batch dim (1,V,3,H,W) to fit your current trainer directly
            imgs_bv = imgs_v.unsqueeze(0)
            return {
                "images_bv": imgs_bv,
                "scene_id": scene_id,
                "frame_ids": frame_ids,
                "image_paths": paths,
                "orig_hw": hw_list,
            }

        elif self.output_mode == "paths":
            # Let wrappers load/resize/normalize
            # We still need H,W for logging (we can read from file on-the-fly or trust ScanNet++ 240x320).
            # To avoid extra I/O here, we record None; wrappers can infer actual size later if needed.
            return {
                "image_paths": paths,
                "scene_id": scene_id,
                "frame_ids": frame_ids,
                "orig_hw": [(None, None)] * len(paths),
            }

        else:
            raise ValueError(f"Unsupported io.output_mode={self.output_mode}")


def scannet_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal collate for B=1; supports B>1 if all frames are same size.
    For PoC, keep batch_size=1 in your DataLoader.
    """
    assert len(batch) > 0
    if "images_bv" in batch[0]:
        # tensor mode
        imgs = torch.cat([b["images_bv"] for b in batch], dim=0)  # (B,V,3,H,W)
        scene_ids = [b["scene_id"] for b in batch]
        frame_ids = [b["frame_ids"] for b in batch]
        image_paths = [b["image_paths"] for b in batch]
        orig_hw = [b["orig_hw"] for b in batch]
        return {
            "images_bv": imgs,
            "scene_id": scene_ids if len(scene_ids) > 1 else scene_ids[0],
            "frame_ids": frame_ids if len(frame_ids) > 1 else frame_ids[0],
            "image_paths": image_paths if len(image_paths) > 1 else image_paths[0],
            "orig_hw": orig_hw if len(orig_hw) > 1 else orig_hw[0],
        }
    else:
        # paths mode
        scene_ids = [b["scene_id"] for b in batch]
        frame_ids = [b["frame_ids"] for b in batch]
        image_paths = [b["image_paths"] for b in batch]
        orig_hw = [b["orig_hw"] for b in batch]
        return {
            "scene_id": scene_ids if len(scene_ids) > 1 else scene_ids[0],
            "frame_ids": frame_ids if len(frame_ids) > 1 else frame_ids[0],
            "image_paths": image_paths if len(image_paths) > 1 else image_paths[0],
            "orig_hw": orig_hw if len(orig_hw) > 1 else orig_hw[0],
        }
