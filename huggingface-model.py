from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

image = Image.open("<Your-image-file-location-here>")
image = to_tensor(image)
image = image.unsqueeze_(0)

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
inputs = feature_extractor(image, return_tensors="pt")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_label = logits.argmax(-1).item()
