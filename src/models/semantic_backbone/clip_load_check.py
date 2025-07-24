import torch
import requests
import yaml
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# --- 1. Load Global Config ---
CONFIG_PATH = "../../../configs/default.yaml"

print(f"Loading configuration from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- 2. Obtain Configs ---
clip_model_path = config['paths']['clip_model_path']
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

# --- 3. Load CLIP Model to Device ---
print(f"Loading model from local path: {clip_model_path}")
clip_model = CLIPModel.from_pretrained(clip_model_path)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
print("Model and processor loaded successfully.")

clip_model.to(device)

# --- 4. Load Images from COCO ---
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog", "a photo of two cats"]

# --- 5. Inference ---
inputs = clip_processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = clip_model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Output Final Score
results = zip(texts, probs.cpu().numpy()[0])
print("\nImage-Text Similarity Probabilities:")
for text, prob in results:
    print(f"- {text}: {prob:.2%}")
