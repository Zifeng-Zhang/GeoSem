import yaml

from src.models.geometry_backbone.vggt_wrapper import VGGTWrapper


with open("../configs/_base/model/vggt.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Option 1: forward from image paths (uses VGGT's load_and_preprocess_images internally)
vggt = VGGTWrapper(cfg['model']['vggt']).eval()
out = vggt.forward_from_paths(["../examples/images/cartoon.jpg"])

print(out['xyz_map'].shape)           # (B,V,H,W,3)
print(out['conf_map'].shape)          # (B,V,H,W)
print(out['patch_tokens_2048'].shape) # (B,V,H_p,W_p,2048)
print(out['camera']['extrinsic'].shape if out['camera']['extrinsic'] is not None else None)
