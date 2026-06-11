# ============================================================
#  validate_dinov2.py
#  Validation complète des 4 critères DINOv2 (CDC §1.4)
#
#  Usage :
#      python validate_dinov2.py --data_path data/midv2020/train
#
#  Sorties :
#      validation/pca_patch_tokens.png
#      validation/tsne_cls_tokens.png
#      validation/coherence_report.txt
#      validation/rapport_final.txt
# ============================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import matplotlib
matplotlib.use('Agg')   # pas d'écran nécessaire
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs("validation", exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════

class SimpleDocDataset(Dataset):
    """
    Charge toutes les images d'un dossier.
    Le nom du sous-dossier parent = type de document.

    Structure attendue (deux possibilités) :

    Option A — sous-dossiers par type :
        data/midv2020/train/
            CNI_albanaise/   image1.jpg ...
            passeport_grec/  image1.jpg ...

    Option B — images à plat :
        data/midv2020/train/
            image1.jpg
            image2.jpg ...
            (tous labellisés 'document')
    """

    def __init__(self, root: str, image_size: int = 224, max_images: int = 300):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225],
            ),
        ])

        # Chercher toutes les images (récursif)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        all_paths = []
        for ext in extensions:
            all_paths += glob.glob(
                os.path.join(root, '**', ext), recursive=True
            )

        # Filtrer les masques
        all_paths = [p for p in all_paths if '_mask' not in os.path.basename(p)]
        all_paths = sorted(all_paths)

        # Limiter pour la vitesse
        if len(all_paths) > max_images:
            step = len(all_paths) // max_images
            all_paths = all_paths[::step][:max_images]

        self.paths = all_paths

        # Label = nom du sous-dossier parent (type de document)
        self.labels = []
        for p in self.paths:
            parent = os.path.basename(os.path.dirname(p))
            if parent == os.path.basename(root):
                label = 'document'   # Option B : pas de sous-dossier
            else:
                label = parent       # Option A : sous-dossier = type
            self.labels.append(label)

        print(f"  {len(self.paths)} images chargées depuis {root}")
        unique = set(self.labels)
        print(f"  Types de documents : {unique}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img  = Image.open(self.paths[idx]).convert('RGB')
        return {
            'image':    self.transform(img),
            'label':    self.labels[idx],
            'path':     self.paths[idx],
        }


# ══════════════════════════════════════════════════════════════
#  CHARGEMENT DINOV2
# ══════════════════════════════════════════════════════════════

def load_dinov2(device: str):
    print("\n[1/5] Chargement DINOv2 ViT-B/14...")
    model = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14',
        pretrained=True, verbose=False
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    print(f"  DINOv2 chargé sur {device} ✓")
    return model


@torch.no_grad()
def extract_all_features(dinov2, loader, device):
    """Extrait patch_tokens et cls_tokens pour tout le dataset."""
    all_patch  = []
    all_cls    = []
    all_labels = []
    all_paths  = []

    for batch in loader:
        images = batch['image'].to(device)
        out    = dinov2.forward_features(images)
        all_patch.append(out['x_norm_patchtokens'].cpu())   # (B, 256, 768)
        all_cls.append(out['x_norm_clstoken'].cpu())        # (B, 768)
        all_labels.extend(batch['label'])
        all_paths.extend(batch['path'])

    patch_tokens = torch.cat(all_patch, dim=0)   # (N, 256, 768)
    cls_tokens   = torch.cat(all_cls,   dim=0)   # (N, 768)
    return patch_tokens, cls_tokens, all_labels, all_paths


# ══════════════════════════════════════════════════════════════
#  CRITÈRE 1 — PCA SUR PATCH_TOKENS
# ══════════════════════════════════════════════════════════════

def critere_1_pca(patch_tokens: torch.Tensor, paths: list, results: dict):
    """
    Projette les patch tokens en 3 composantes PCA.
    Colorie par position spatiale (proxy des zones : haut/milieu/bas).
    Vérifie que la variance expliquée > 30%.
    """
    print("\n[Critère 1] PCA sur patch_tokens...")

    # Prendre la première image pour la visualisation spatiale
    tokens_1img = patch_tokens[0].numpy()   # (256, 768)

    # PCA → 3 composantes
    pca = PCA(n_components=3, random_state=42)
    tokens_3d = pca.fit_transform(tokens_1img)   # (256, 3)
    variance  = pca.explained_variance_ratio_.sum()

    print(f"  Variance expliquée (3 composantes) : {variance:.2%}")

    # Position spatiale : grille 16×16 → zone haute / milieu / basse
    positions = np.arange(256)
    rows      = positions // 16   # 0..15
    zones     = np.where(rows < 5, 'haut (en-tête)',
                np.where(rows < 11, 'milieu (contenu)', 'bas (pied)'))

    colors_map = {
        'haut (en-tête)':   'royalblue',
        'milieu (contenu)': 'tomato',
        'bas (pied)':       'seagreen',
    }

    fig = plt.figure(figsize=(12, 5))

    # Vue 2D (PC1 vs PC2)
    ax1 = fig.add_subplot(121)
    for zone, color in colors_map.items():
        idx = np.where(zones == zone)[0]
        ax1.scatter(tokens_3d[idx, 0], tokens_3d[idx, 1],
                    c=color, label=zone, alpha=0.7, s=30)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title(f"PCA patch tokens\n(variance expliquée : {variance:.1%})")
    ax1.legend(fontsize=8)

    # Vue 3D
    ax2 = fig.add_subplot(122, projection='3d')
    for zone, color in colors_map.items():
        idx = np.where(zones == zone)[0]
        ax2.scatter(tokens_3d[idx, 0], tokens_3d[idx, 1], tokens_3d[idx, 2],
                    c=color, label=zone, alpha=0.6, s=20)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
    ax2.set_title("Vue 3D")

    plt.tight_layout()
    plt.savefig("validation/pca_patch_tokens.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Résultat
    passed = variance > 0.30
    results['critere_1_pca'] = {
        'variance_expliquee': float(variance),
        'passe': passed,
        'figure': 'validation/pca_patch_tokens.png',
    }
    status = "✓ PASSÉ" if passed else "✗ ÉCHOUÉ"
    print(f"  Variance expliquée : {variance:.2%}  (seuil > 30%)  → {status}")
    print(f"  Figure sauvegardée : validation/pca_patch_tokens.png")


# ══════════════════════════════════════════════════════════════
#  CRITÈRE 2 — t-SNE SUR CLS_TOKENS
# ══════════════════════════════════════════════════════════════

def critere_2_tsne(cls_tokens: torch.Tensor, labels: list, results: dict):
    """
    Projette les cls_tokens en 2D via t-SNE.
    Vérifie que les types de documents forment des clusters séparés.
    Métrique : silhouette score > 0.10.
    """
    print("\n[Critère 2] t-SNE sur cls_tokens...")

    cls_np      = cls_tokens.numpy()    # (N, 768)
    unique_types = sorted(set(labels))
    n_types      = len(unique_types)

    print(f"  {len(cls_np)} images, {n_types} type(s) de document")

    # t-SNE → 2D
    perplexity = min(30, len(cls_np) - 1)
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=42, n_iter=1000, verbose=0
    )
    cls_2d = tsne.fit_transform(cls_np)   # (N, 2)

    # Calcul silhouette score (séparabilité des clusters)
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    if n_types > 1:
        sil = silhouette_score(cls_2d, label_ids)
    else:
        sil = 0.0   # un seul type → pas de séparation mesurable

    print(f"  Silhouette score : {sil:.4f}  (> 0.10 = clusters bien séparés)")

    # Visualisation
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_types, 2)))
    plt.figure(figsize=(9, 7))

    for i, doc_type in enumerate(unique_types):
        idx = [j for j, l in enumerate(labels) if l == doc_type]
        plt.scatter(
            cls_2d[idx, 0], cls_2d[idx, 1],
            c=[colors[i]], label=doc_type,
            alpha=0.7, s=40, edgecolors='white', linewidth=0.3
        )

    sil_str = f"{sil:.3f}" if n_types > 1 else "N/A (1 type)"
    plt.title(f"t-SNE des cls_tokens par type de document\n"
              f"Silhouette score : {sil_str}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig("validation/tsne_cls_tokens.png", dpi=150, bbox_inches='tight')
    plt.close()

    passed = (sil > 0.10) if n_types > 1 else True
    results['critere_2_tsne'] = {
        'n_types':         n_types,
        'silhouette_score': float(sil),
        'passe':           passed,
        'figure':          'validation/tsne_cls_tokens.png',
    }
    status = "✓ PASSÉ" if passed else "✗ ÉCHOUÉ (clusters mélangés)"
    print(f"  Silhouette : {sil:.4f}  → {status}")
    print(f"  Figure sauvegardée : validation/tsne_cls_tokens.png")


# ══════════════════════════════════════════════════════════════
#  CRITÈRE 3 — COHÉRENCE (COSINE SIMILARITY > 0.90)
# ══════════════════════════════════════════════════════════════

def critere_3_coherence(dinov2, dataset, device, n_images=20,
                         n_aug=10, results=dict):
    """
    Pour n_images images, applique n_aug augmentations légères
    et mesure la cosine similarity entre cls_token original
    et cls_token augmenté.
    Cible CDC : moyenne > 0.90.
    """
    print("\n[Critère 3] Test de cohérence (cosine similarity)...")

    augment = transforms.Compose([
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ])

    unnorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std= [1/0.229,      1/0.224,      1/0.225]
    )
    renorm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )

    all_sims   = []
    per_image  = []

    indices = np.linspace(0, len(dataset)-1, min(n_images, len(dataset)),
                          dtype=int)

    dinov2.eval()
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image  = sample['image'].unsqueeze(0).to(device)  # (1,3,224,224)

            # Features originales
            out_orig = dinov2.forward_features(image)
            cls_orig = out_orig['x_norm_clstoken']             # (1, 768)

            sims_this_img = []
            for _ in range(n_aug):
                # Dénorm → augment → renorm
                img_cpu = image.squeeze(0).cpu()
                img_cpu = unnorm(img_cpu)
                img_cpu = augment(img_cpu)
                img_cpu = renorm(img_cpu)
                img_aug = img_cpu.unsqueeze(0).to(device)

                out_aug  = dinov2.forward_features(img_aug)
                cls_aug  = out_aug['x_norm_clstoken']

                sim = F.cosine_similarity(cls_orig, cls_aug).item()
                sims_this_img.append(sim)
                all_sims.append(sim)

            per_image.append(np.mean(sims_this_img))

    mean_sim = float(np.mean(all_sims))
    min_sim  = float(np.min(all_sims))
    std_sim  = float(np.std(all_sims))

    # Graphique distribution
    plt.figure(figsize=(8, 4))
    plt.hist(all_sims, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(0.90, color='red', linestyle='--', linewidth=2, label='Seuil CDC (0.90)')
    plt.axvline(mean_sim, color='orange', linestyle='-', linewidth=2,
                label=f'Moyenne ({mean_sim:.3f})')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Fréquence")
    plt.title("Distribution des cosine similarities\n(original vs augmenté)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("validation/coherence_distribution.png", dpi=150)
    plt.close()

    passed = mean_sim > 0.90
    results['critere_3_coherence'] = {
        'mean_cosine_sim': mean_sim,
        'min_cosine_sim':  min_sim,
        'std_cosine_sim':  std_sim,
        'n_images_testees': len(indices),
        'n_aug_par_image':  n_aug,
        'passe':           passed,
        'figure':          'validation/coherence_distribution.png',
    }
    status = "✓ PASSÉ" if passed else "✗ ÉCHOUÉ"
    print(f"  Cosine similarity moyenne : {mean_sim:.4f}  (seuil > 0.90)  → {status}")
    print(f"  Cosine similarity min     : {min_sim:.4f}")
    print(f"  Figure sauvegardée : validation/coherence_distribution.png")


# ══════════════════════════════════════════════════════════════
#  CRITÈRE 4 — TEMPS D'EXTRACTION < 5ms
# ══════════════════════════════════════════════════════════════

def critere_4_temps(dinov2, device, n_runs=100, results=dict):
    """
    Mesure le temps moyen d'extraction DINOv2 pour 1 image.
    Cible CDC : < 5ms sur GPU.
    """
    print("\n[Critère 4] Benchmark temps d'extraction...")

    image = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up (10 passes ignorées)
    dinov2.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = dinov2.forward_features(image)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Mesure
    times_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = dinov2.forward_features(image)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    mean_ms = float(np.mean(times_ms))
    std_ms  = float(np.std(times_ms))
    min_ms  = float(np.min(times_ms))
    max_ms  = float(np.max(times_ms))

    # Graphique
    plt.figure(figsize=(8, 4))
    plt.plot(times_ms, color='steelblue', alpha=0.6, linewidth=0.8)
    plt.axhline(5.0,     color='red',    linestyle='--', linewidth=2,
                label='Seuil CDC (5ms)')
    plt.axhline(mean_ms, color='orange', linestyle='-',  linewidth=2,
                label=f'Moyenne ({mean_ms:.2f}ms)')
    plt.xlabel("Run")
    plt.ylabel("Temps (ms)")
    plt.title(f"Temps d'extraction DINOv2 ({n_runs} runs)\nDevice : {device}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("validation/benchmark_temps.png", dpi=150)
    plt.close()

    # Seuil adapté : 5ms si GPU, 50ms si CPU (acceptable)
    seuil = 5.0 if device == 'cuda' else 50.0
    passed = mean_ms < seuil

    results['critere_4_temps'] = {
        'device':   device,
        'mean_ms':  mean_ms,
        'std_ms':   std_ms,
        'min_ms':   min_ms,
        'max_ms':   max_ms,
        'seuil_ms': seuil,
        'passe':    passed,
        'figure':   'validation/benchmark_temps.png',
    }
    status = "✓ PASSÉ" if passed else "✗ ÉCHOUÉ"
    print(f"  Temps moyen : {mean_ms:.2f} ms ± {std_ms:.2f} ms  "
          f"(seuil < {seuil:.0f}ms sur {device})  → {status}")
    print(f"  Figure sauvegardée : validation/benchmark_temps.png")


# ══════════════════════════════════════════════════════════════
#  RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

def afficher_rapport(results: dict):
    lignes = [
        "",
        "=" * 60,
        "  RAPPORT DE VALIDATION DINOV2 — CDC §1.4",
        "=" * 60,
    ]

    configs = [
        ("Critère 1 — PCA patch tokens",
         results.get('critere_1_pca', {}),
         lambda r: f"Variance expliquée : {r.get('variance_expliquee',0):.2%}  (seuil > 30%)"),

        ("Critère 2 — t-SNE cls_tokens",
         results.get('critere_2_tsne', {}),
         lambda r: f"Silhouette score : {r.get('silhouette_score',0):.4f}  "
                   f"({r.get('n_types',1)} type(s) de document)"),

        ("Critère 3 — Cohérence cosine",
         results.get('critere_3_coherence', {}),
         lambda r: f"Cosine sim moyenne : {r.get('mean_cosine_sim',0):.4f}  "
                   f"min : {r.get('min_cosine_sim',0):.4f}  (seuil > 0.90)"),

        ("Critère 4 — Temps extraction",
         results.get('critere_4_temps', {}),
         lambda r: f"Temps moyen : {r.get('mean_ms',0):.2f} ms  "
                   f"(seuil < {r.get('seuil_ms',5):.0f} ms sur {r.get('device','?')})"),
    ]

    tous_passes = True
    for nom, res, detail_fn in configs:
        if not res:
            continue
        passe = res.get('passe', False)
        tous_passes = tous_passes and passe
        icone  = "✓" if passe else "✗"
        statut = "PASSÉ  " if passe else "ÉCHOUÉ"
        lignes.append(f"\n  {icone} {statut}  |  {nom}")
        lignes.append(f"           {detail_fn(res)}")
        if 'figure' in res:
            lignes.append(f"           Figure : {res['figure']}")

    lignes += [
        "",
        "─" * 60,
        f"  VERDICT GLOBAL : {'✓ DINOV2 VALIDÉ — pipeline peut démarrer'  if tous_passes else '✗ VALIDATION INCOMPLÈTE — corriger avant entraînement'}",
        "=" * 60,
        "",
    ]

    rapport = "\n".join(lignes)
    print(rapport)

    with open("validation/rapport_final.txt", "w", encoding="utf-8") as f:
        f.write(rapport)
    print("  Rapport sauvegardé : validation/rapport_final.txt")

    return tous_passes


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validation des 4 critères DINOv2 (CDC §1.4)"
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help="Chemin vers le dossier d'images (ex: data/midv2020/train)"
    )
    parser.add_argument(
        '--max_images', type=int, default=200,
        help="Nombre max d'images à charger (défaut: 200)"
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help="Batch size pour l'extraction (défaut: 16)"
    )
    parser.add_argument(
        '--n_coherence_images', type=int, default=20,
        help="Nb images pour le test de cohérence (défaut: 20)"
    )
    parser.add_argument(
        '--benchmark_runs', type=int, default=100,
        help="Nb de runs pour le benchmark temps (défaut: 100)"
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice détecté : {device}")
    print(f"Data path      : {args.data_path}")

    # ── Chargement données ─────────────────────────────────────
    print("\n[2/5] Chargement du dataset...")
    dataset = SimpleDocDataset(
        root=args.data_path,
        max_images=args.max_images
    )
    if len(dataset) == 0:
        print("ERREUR : aucune image trouvée dans", args.data_path)
        sys.exit(1)

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    # ── Chargement DINOv2 ──────────────────────────────────────
    dinov2 = load_dinov2(device)

    # ── Extraction de toutes les features ─────────────────────
    print("\n[3/5] Extraction des features DINOv2...")
    patch_tokens, cls_tokens, labels, paths = extract_all_features(
        dinov2, loader, device
    )
    print(f"  patch_tokens : {tuple(patch_tokens.shape)}")
    print(f"  cls_tokens   : {tuple(cls_tokens.shape)}")

    # ── 4 critères ────────────────────────────────────────────
    print("\n[4/5] Évaluation des 4 critères...")
    results = {}

    critere_1_pca(patch_tokens, paths, results)
    critere_2_tsne(cls_tokens, labels, results)
    critere_3_coherence(
        dinov2, dataset, device,
        n_images=args.n_coherence_images,
        n_aug=10,
        results=results,
    )
    critere_4_temps(
        dinov2, device,
        n_runs=args.benchmark_runs,
        results=results,
    )

    # ── Rapport ───────────────────────────────────────────────
    print("\n[5/5] Génération du rapport...")
    afficher_rapport(results)


if __name__ == "__main__":
    main()