import torch
import yaml

from third_party.vggt.vggt.models.vggt import VGGT
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images


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
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        print(predictions)
