"""Divisao treino/validacao/teste por IDENTIDADE e checagem de data leakage (topicos 29 e 30)."""
import hashlib

import numpy as np


def split_identities(labels, fractions=(0.6, 0.2, 0.2), seed=42):
    """Cada pessoa cai em UM unico conjunto. Assim nenhuma foto (nem foto parecida) de uma
    pessoa do teste aparece no treino ou na validacao."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    ids = np.unique(labels)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    parts = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    return {name: np.flatnonzero(np.isin(labels, p)) for name, p in parts.items()}


def check_no_leakage(split_indices, labels, images=None):
    labels = np.asarray(labels)
    names = list(split_indices)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(labels[split_indices[a]]) & set(labels[split_indices[b]])
            if shared:
                raise AssertionError(f"Identidades em '{a}' e '{b}' ao mesmo tempo: {sorted(shared)[:5]}")
    if images is not None:  # imagens identicas (byte a byte) em conjuntos diferentes
        seen = {}
        for name, idx in split_indices.items():
            for j in idx:
                h = hashlib.sha1(images[j].tobytes()).hexdigest()
                if h in seen and seen[h] != name:
                    raise AssertionError(f"Imagem repetida em '{seen[h]}' e '{name}'")
                seen[h] = name
    return True
