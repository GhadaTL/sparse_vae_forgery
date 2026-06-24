"""
data/datasets.py
Datasets PyTorch pour MIDV-2020 (train/val — authentiques uniquement)
et FMIDV-2022 (test — authentiques + forgés).

IMPORTANT : Les forgeries ne doivent JAMAIS apparaître dans les loaders
d'entraînement ou de validation.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


# =============================================================================
# Transformations
# =============================================================================

def get_train_transform(image_size: int = 224) -> T.Compose:
    """
    Augmentations légères uniquement — les documents ont une orientation fixe.
    Pas de flip vertical, pas de rotation forte.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.RandomHorizontalFlip(p=0.3),   # flip horizontal léger autorisé
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Transformation d'évaluation — déterministe."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# Dataset MIDV-2020 — Authentiques uniquement
# =============================================================================

class MIDV2020Dataset(Dataset):
    """
    Dataset MIDV-2020 pour l'entraînement et la validation.
    Ne charge QUE les documents authentiques.

    Structure attendue :
        midv2020_root/
            <country_code>/
                <doc_id>/
                    images/
                        frame0000.jpg
                        ...
    """

    def __init__(self,
                 root: str,
                 transform: Optional[T.Compose] = None,
                 image_size: int = 224):
        self.root = Path(root)
        self.transform = transform or get_eval_transform(image_size)
        self.image_paths = self._collect_images()

        if len(self.image_paths) == 0:
            raise ValueError(
                f"Aucune image trouvée dans {root}. "
                "Vérifiez la structure du dossier MIDV-2020."
            )
        print(f"[MIDV-2020] {len(self.image_paths)} images authentiques chargées.")

    def _collect_images(self) -> List[Path]:
        """Collecte récursivement toutes les images .jpg/.png."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = []
        for ext in extensions:
            paths.extend(self.root.rglob(f"*{ext}"))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "path": str(path),
            "label": 0,   # 0 = authentique
        }


# =============================================================================
# Dataset FMIDV-2022 — Authentiques + Forgés (TEST ONLY)
# =============================================================================

class FMIDV2022Dataset(Dataset):
    """
    Dataset FMIDV-2022 pour l'évaluation finale uniquement.
    Contient documents authentiques (label=0) et forgés (label=1).

    Structure attendue :
        fmidv2022_root/
            authentic/
                <images>.jpg
            forged/
                <images>.jpg
    OU structure plate avec un fichier CSV d'annotations.
    """

    def __init__(self,
                 root: str,
                 transform: Optional[T.Compose] = None,
                 image_size: int = 224):
        self.root = Path(root)
        self.transform = transform or get_eval_transform(image_size)
        self.samples = self._collect_samples()

        n_auth = sum(1 for _, l in self.samples if l == 0)
        n_forg = sum(1 for _, l in self.samples if l == 1)
        print(f"[FMIDV-2022] {n_auth} authentiques, {n_forg} forgés.")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Collecte les images avec labels.
        Cherche les sous-dossiers 'authentic' et 'forged'.
        Fallback : cherche un fichier labels.csv.
        """
        samples = []
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        auth_dir = self.root / "authentic"
        forg_dir = self.root / "forged"

        if auth_dir.exists():
            for ext in extensions:
                for p in sorted(auth_dir.rglob(f"*{ext}")):
                    samples.append((p, 0))

        if forg_dir.exists():
            for ext in extensions:
                for p in sorted(forg_dir.rglob(f"*{ext}")):
                    samples.append((p, 1))

        # Fallback CSV
        if not samples:
            csv_path = self.root / "labels.csv"
            if csv_path.exists():
                import csv
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        path = self.root / row["filename"]
                        label = int(row["label"])
                        samples.append((path, label))

        if not samples:
            raise ValueError(
                f"Aucune image trouvée dans {self.root}. "
                "Attendu : sous-dossiers 'authentic/' et 'forged/'."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "path": str(path),
            "label": label,
        }


# =============================================================================
# Factories de DataLoaders
# =============================================================================

def get_midv2020_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders train/val à partir de MIDV-2020.
    Retourne (train_loader, val_loader).
    """
    full_dataset = MIDV2020Dataset(
        root=cfg.data.midv2020_root,
        transform=None,  # On assignera les transforms après le split
        image_size=cfg.data.image_size,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * cfg.data.train_split)
    n_val   = n_total - n_train

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    # Assigner les transforms différentes
    train_ds.dataset.transform = get_train_transform(cfg.data.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    print(f"[DataLoaders] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


def get_fmidv2022_loader(cfg) -> DataLoader:
    """Crée le DataLoader de test FMIDV-2022."""
    dataset = FMIDV2022Dataset(
        root=cfg.data.fmidv2022_root,
        image_size=cfg.data.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
