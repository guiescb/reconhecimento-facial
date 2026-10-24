"""Carregamento do dataset publico (LFW, variante funneled) via scikit-learn.

O fetch_lfw_people baixa uma copia do arquivo, hospedada no figshare, para a pasta do
scikit-learn (fora do repositorio), entao nada do dataset e redistribuido no GitHub.
Fonte oficial: http://vis-www.cs.umass.edu/lfw/
"""

import numpy as np


def load_lfw(min_faces_per_person=10, crop=144, image_size=160):
    from PIL import Image
    from sklearn.datasets import fetch_lfw_people

    lo = (250 - crop) // 2
    hi = lo + crop
    ds = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        color=True,
        resize=1.0,
        slice_=(slice(lo, hi), slice(lo, hi)),
    )
    imgs = ds.images
    if imgs.dtype != np.uint8:
        scale = 255.0 if imgs.max() <= 1.0 else 1.0
        imgs = np.clip(imgs * scale, 0, 255).astype(np.uint8)
    resized = np.stack(
        [np.asarray(Image.fromarray(a).resize((image_size, image_size), Image.BILINEAR)) for a in imgs]
    )
    return resized, ds.target.astype(int), list(ds.target_names)


def synthetic_embeddings(n_ids=60, min_imgs=8, max_imgs=15, dim=512, noise=3.0, seed=42):
    """Embeddings sinteticos (sem rostos reais) para testar o pipeline sem baixar nada."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_ids, dim))
    embs, labels = [], []
    for c in range(n_ids):
        n = int(rng.integers(min_imgs, max_imgs + 1))
        embs.append(centers[c] + noise * rng.normal(size=(n, dim)))
        labels += [c] * n
    emb = np.concatenate(embs)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype(np.float32), np.array(labels)
