import torch
from PIL import Image
import torchvision.transforms as T

# =========================
# 1. Charger DINOv2
# =========================
print("Loading DINOv2...")

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

model.eval()

# =========================
# 2. Image input
# =========================
img_path = "test.jpg"   # <-- mets ton image ici

try:
    image = Image.open(img_path).convert("RGB")
except Exception as e:
    print("Erreur de chargement image:", e)
    exit()

print("Image chargée:", img_path)

# =========================
# 3. Preprocessing
# =========================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

x = transform(image).unsqueeze(0)  # batch size = 1

# =========================
# 4. Forward pass
# =========================
with torch.no_grad():
    output = model(x)

# =========================
# 5. Results
# =========================
print("\n===== OUTPUT =====")
print("Shape:", output.shape)
print("First values:", output[0][:10])
print("==================")