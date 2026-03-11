from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Configuration
# ----------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dossier de la banque d'images
DATABASE_DIR = Path("C:/Users/tomde/python-apps/vincimap/data/TEST_002/images")

# Image requête
QUERY_IMAGE = Path("C:/Users/tomde/python-apps/vincimap/data/image.png")

# Nombre de résultats à retourner
TOP_K = 5

# ----------------------------
# Modèle de features
# ----------------------------
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# On enlève la dernière couche de classification
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
feature_extractor.eval()

# Transformations associées aux poids du modèle
preprocess = weights.transforms()


# ----------------------------
# Fonctions utilitaires
# ----------------------------
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


@torch.no_grad()
def extract_embedding(image_path: Path) -> np.ndarray:
    """Charge une image et retourne son embedding normalisé."""
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)   # [1, C, H, W]
    feats = feature_extractor(x)                  # [1, 2048, 1, 1]
    feats = feats.flatten(1)                      # [1, 2048]
    feats = F.normalize(feats, p=2, dim=1)       # normalisation L2
    return feats.squeeze(0).cpu().numpy()


def build_database_embeddings(database_dir: Path):
    """Construit la liste des embeddings pour toutes les images du dossier."""
    image_paths = [p for p in database_dir.rglob("*") if p.is_file() and is_image_file(p)]

    if not image_paths:
        raise ValueError(f"Aucune image trouvée dans {database_dir.resolve()}")

    embeddings = []
    valid_paths = []

    for path in image_paths:
        try:
            emb = extract_embedding(path)
            embeddings.append(emb)
            valid_paths.append(path)
        except Exception as e:
            print(f"[IGNORÉ] {path} -> {e}")

    if not embeddings:
        raise ValueError("Impossible d'extraire des embeddings sur les images de la base.")

    return valid_paths, np.vstack(embeddings)


def search_similar_images(query_image: Path, db_paths, db_embeddings, top_k=5):
    """Compare l'image requête à la base et retourne les top résultats."""
    query_emb = extract_embedding(query_image).reshape(1, -1)

    sims = cosine_similarity(query_emb, db_embeddings)[0]  # scores de similarité
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        similarity = float(sims[idx])
        percentage = max(0.0, min(100.0, similarity * 100))  # score affiché en %
        results.append({
            "image": str(db_paths[idx]),
            "score": similarity,
            "percentage": percentage
        })
    return results


if __name__ == "__main__":
    db_paths, db_embeddings = build_database_embeddings(DATABASE_DIR)
    results = search_similar_images(QUERY_IMAGE, db_paths, db_embeddings, top_k=TOP_K)

    print(f"\nRésultats les plus proches de : {QUERY_IMAGE}\n")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item['image']} | "
            f"similarité = {item['score']:.4f} | "
            f"correspondance = {item['percentage']:.2f}%"
        )