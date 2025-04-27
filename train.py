import os

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)


class StickerDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label


stickers = StickerDataset("data/stickers", 1, transform=preprocess)
non_stickers = StickerDataset("data/non_stickers", 0, transform=preprocess)

full_dataset = torch.utils.data.ConcatDataset([stickers, non_stickers])
train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=2)


class CLIPHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.fc(x)


classifier = CLIPHead().to(device)

optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

clip_model.eval()
for epoch in range(200):
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            features = clip_model.encode_image(imgs)
            features = features / features.norm(dim=1, keepdim=True)
            features = features.float()

        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Accuracy={correct/total:.4f}"
    )

# Save the classifier head
os.makedirs("models", exist_ok=True)
torch.save(classifier.state_dict(), "models/clip_classifier.pth")
print("Model saved to models/clip_classifier.pth")
