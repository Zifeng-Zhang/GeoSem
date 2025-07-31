import torch
import yaml

from third_party.vggt.vggt.models.vggt import VGGT
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from third_party.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from third_party.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map


# Load Global Config
CONFIG_PATH = "../../../configs/default.yaml"

print(f"Loading configuration from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Obtain Configs
vggt_path = config['paths']['vggt_model_path']
device_setting = config['general']['device']
random_seed = config['general']['random_seed']

# set random seed
torch.manual_seed(random_seed)

# set CUDA option
if device_setting == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_setting

print(f"Using device: {device}")

# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Load and preprocess example images (replace with your own image paths)
image_names = ["../../../examples/images/cartoon.jpg"]
images = load_and_preprocess_images(image_names).to(device)

# Initialize the model and load the pretrained weights.
# Since VGGT .pt once downloaded before, the program should directly load it.
model = VGGT.from_pretrained(vggt_path).to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)

    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                 extrinsic.squeeze(0),
                                                                 intrinsic.squeeze(0))
