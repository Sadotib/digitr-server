from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import rotate, shift
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # (B, 32, 26, 26)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3),  # (B, 64, 24, 24)
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def preprocess_image(path):
    # 1. Load image and convert to grayscale
    img = Image.open(path).convert("L")

    # 2. Invert (MNIST style: white digit on black)
    img = ImageOps.invert(img)

    # 3. Moderate contrast (NOT extreme)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(5.0)

    # 4. Light denoising
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # 5. Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # 6. Convert to NumPy and normalize
    arr = np.array(img, dtype=np.float32) / 255.0

    # 7. Optional gentle cleanup (keep gradients)
    arr[arr < 0.45] = 0.0

    # 8. Reshape for CNN: (1, 1, 28, 28)
    arr = arr.reshape(1, 1, 28, 28)

    # 9. Convert to torch tensor
    return torch.tensor(arr, dtype=torch.float32)

def predict_single_image(path, model):
    model.eval()

    X = preprocess_image(path).to(device)

    with torch.no_grad():
        output = model(X)
        pred = output.argmax(dim=1).item()

    # # Visualize
    # img = X[0, 0].cpu().numpy()
    # plt.imshow(img, cmap="gray")
    # plt.title(f"Prediction: {pred}")
    # plt.axis("off")
    # plt.show()

    return pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("digitR_cnn.pth", map_location=device))
model.eval()
path = "23.jpg"
pred = predict_single_image(path, model)
print(pred)