import os

import clip
import torch
import torch.nn as nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()


class CLIPHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.fc(x)


# Load the trained classifier
classifier = CLIPHead().to(device)
classifier.load_state_dict(
    torch.load("models/clip_classifier.pth", map_location=device)
)
classifier.eval()

print("Model Loaded!")


def predict_sticker(img_path: str) -> float:
    img = Image.open(img_path).convert("RGB")
    img_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = clip_model.encode_image(img_input)
        features = features / features.norm(dim=1, keepdim=True)
        features = features.float()

        logits = classifier(features)
        probs = torch.softmax(logits, dim=1)

    return probs[0, 1].item()


if __name__ == "__main__":
    test_images = [
        "uploads/test_images/non_stickers",
        "uploads/test_images/stickers",
    ]
    for folder in test_images:
        print(f"Testing folder: {folder}")
        for filename in os.listdir(folder):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                prob = predict_sticker(img_path)
                print(f"{img_path}: {prob:.4f}")
