import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Literal, Tuple


# ── Transform standard CDC §5.1 ───────────────────────────────────────────────
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


class MIDV2020Dataset(Dataset):
    """
    MIDV-2020 — documents authentiques.

    Structure attendue sur disque :
        data/midv2020/
            train/   *.jpg / *.png
            val/     *.jpg / *.png

    Label : 0 (authentique) pour tous les fichiers.
    """

    def __init__(self,
                 split: Literal["train", "val"] = "train",
                 root_dir: str = "data/midv2020",
                 transform=None):

        self.root_dir  = Path(root_dir) / split
        self.transform = transform or DEFAULT_TRANSFORM
        self.label     = 0

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dossier introuvable : {self.root_dir}\n"
                f"Crée la structure : data/midv2020/train/ et data/midv2020/val/"
            )

        self.image_paths = sorted(
            list(self.root_dir.glob("*.jpg")) +
            list(self.root_dir.glob("*.jpeg")) +
            list(self.root_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.label


class FMIDV2022Dataset(Dataset):
    """
    FMIDV-2022 — documents forgés.

    Structure attendue sur disque :
        data/fmidv2022/
            *.jpg / *.png

    Label : 1 (forgé) pour tous les fichiers.
    """

    def __init__(self,
                 root_dir: str = "data/fmidv2022",
                 transform=None):

        self.root_dir  = Path(root_dir)
        self.transform = transform or DEFAULT_TRANSFORM
        self.label     = 1

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dossier introuvable : {self.root_dir}\n"
                f"Crée la structure : data/fmidv2022/"
            )

        self.image_paths = sorted(
            list(self.root_dir.glob("*.jpg")) +
            list(self.root_dir.glob("*.jpeg")) +
            list(self.root_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.label
