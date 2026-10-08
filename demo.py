# Usage: python demo.py path/to/bin_photo.jpg [more.jpg ...]
import sys, torch, torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models

ck = torch.load("model.pt", map_location="cpu")
model = getattr(models, ck["arch"])(weights=None)
model.fc = nn.Linear(model.fc.in_features, ck["num_classes"])
model.load_state_dict(ck["state_dict"]); model.eval()

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def predict(path):
    img = Image.open(path).convert("RGB").resize((ck["img"], ck["img"]), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model((x - mean) / std), 1)[0]
    return int(probs.argmax()), probs

if __name__ == "__main__":
    for path in sys.argv[1:]:
        c, pr = predict(path)
        print(f"{path}: predicted {c} item(s)  (confidence {pr[c]:.0%})")
