"""Extracao de embeddings com FaceNet (InceptionResnetV1) pre-treinado, em CPU."""
from pathlib import Path

import numpy as np


def extract_embeddings(images_uint8, batch_size=32, device="cpu", pretrained="vggface2"):
    import torch
    from facenet_pytorch import InceptionResnetV1
    from tqdm import tqdm

    model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images_uint8), batch_size), desc="embeddings"):
            x = torch.from_numpy(images_uint8[i:i + batch_size]).permute(0, 3, 1, 2).float()
            x = (x - 127.5) / 128.0  # mesma padronizacao usada no treino do modelo
            e = torch.nn.functional.normalize(model(x.to(device)), dim=1)
            out.append(e.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def cached_embeddings(cache_path, compute_fn):
    """Calcula uma vez e guarda em disco; nas proximas execucoes so le o arquivo."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)["emb"]
    emb = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb)
    return emb
