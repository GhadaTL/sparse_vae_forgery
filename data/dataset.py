# data/dataset.py
# CDC § Datasets — Document Forgery Detection
# À compléter : MIDV2020Dataset, FMIDV2022Dataset, etc.

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Literal


class MIDV2020Dataset(Dataset):
    """
    MIDV-2020 Dataset — Images de documents authentiques.
    
    Splits:
    - train (70%): Entraînement
    - val (30%): Validation et calibration AnomalyScorer
    
    À IMPLÉMENTER par l'utilisateur.
    """
    
    def __init__(self, split: Literal['train', 'val', 'test'] = 'train',
                 root_dir: str = 'data/midv2020',
                 transform=None):
        """
        Args:
            split : 'train', 'val', ou 'test'
            root_dir : chemin vers data/midv2020/
            transform : transformations optionnelles (resize, normalize, etc.)
        """
        self.split = split
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # À IMPLÉMENTER : charger les images et labels
        raise NotImplementedError(
            "MIDV2020Dataset.__init__ — À implémenter avec chargement des images"
        )
    
    def __len__(self) -> int:
        """À IMPLÉMENTER"""
        raise NotImplementedError("MIDV2020Dataset.__len__")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        À IMPLÉMENTER
        
        Returns:
            image : torch.Tensor (3, 224, 224) — normalisée [0,1]
            label : int — 0 (authentique)
        """
        raise NotImplementedError("MIDV2020Dataset.__getitem__")


class FMIDV2022Dataset(Dataset):
    """
    FMIDV-2022 Dataset — Images de documents avec manipulations (forgeries).
    
    Utilisé pour:
    - Test AUC-ROC (evaluate.py)
    - Évaluation AnomalyScorer
    
    À IMPLÉMENTER par l'utilisateur.
    """
    
    def __init__(self, root_dir: str = 'data/fmidv2022', transform=None):
        """
        Args:
            root_dir : chemin vers data/fmidv2022/
            transform : transformations optionnelles
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # À IMPLÉMENTER : charger les images forgées et labels
        raise NotImplementedError(
            "FMIDV2022Dataset.__init__ — À implémenter avec chargement des images"
        )
    
    def __len__(self) -> int:
        """À IMPLÉMENTER"""
        raise NotImplementedError("FMIDV2022Dataset.__len__")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        À IMPLÉMENTER
        
        Returns:
            image : torch.Tensor (3, 224, 224) — normalisée [0,1]
            label : int — 1 (forgé)
        """
        raise NotImplementedError("FMIDV2022Dataset.__getitem__")


class FantasyIDDataset(Dataset):
    """
    FantasyID Dataset — Cross-dataset evaluation.
    
    À IMPLÉMENTER si nécessaire pour ablation croisée.
    """
    
    def __init__(self, root_dir: str = 'data/fantasyid', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        raise NotImplementedError("FantasyIDDataset — À implémenter")
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError()


class SIDTDDataset(Dataset):
    """
    SIDTD Dataset — Cross-dataset evaluation.
    
    À IMPLÉMENTER si nécessaire pour ablation croisée.
    """
    
    def __init__(self, root_dir: str = 'data/sidtd', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        raise NotImplementedError("SIDTDDataset — À implémenter")
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError()
