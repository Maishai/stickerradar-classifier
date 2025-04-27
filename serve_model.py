import base64
import io
import os

import clip
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()


class CLIPHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.fc(x)


# Load trained classifier weights
classifier = CLIPHead().to(device)
classifier.load_state_dict(
    torch.load("models/clip_classifier.pth", map_location=device)
)
classifier.eval()


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


class ImageRequest(BaseModel):
    image_base64: str


def predict_sticker_image(image: Image.Image) -> float:
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img_input)
        features = features / features.norm(dim=1, keepdim=True)
        logits = classifier(features.float())
        probs = torch.softmax(logits, dim=1)
    return probs[0, 1].item()


@app.post("/predict")
async def predict(req: ImageRequest, api_key: str = Depends(verify_api_key)):
    try:
        image_data = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    prob = predict_sticker_image(image)
    return {"sticker_probability": prob}

