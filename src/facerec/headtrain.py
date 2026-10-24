"""E03 - Treino de uma camada de projecao sobre embeddings CONGELADOS do FaceNet.

O FaceNet nao e alterado. Treinamos so uma camada linear (512 -> 512) por cima dos embeddings
ja calculados, usando apenas as identidades de TREINO (topicos 28 e 29). A camada comeca como a
matriz identidade, entao a epoca 0 e exatamente o baseline do E01/E02.

Duas funcoes de perda (topico 28):
- "triplet": Triplet Loss com mineracao batch-hard (metric learning).
- "cross_entropy": classificacao das identidades de treino com classificador por cosseno
  (Cross-Entropy sobre logits = escala * cosseno).

A epoca final e escolhida pelo menor EER de pares na VALIDACAO (nunca no teste).
"""
import copy

import numpy as np

from facerec.metrics import pair_scores, roc_eer


def pk_batch(by_class, P, K, rng):
    """Sorteia P identidades e K fotos de cada uma (necessario para a Triplet Loss)."""
    classes = list(by_class)
    chosen = rng.choice(len(classes), size=min(P, len(classes)), replace=False)
    idx = []
    for ci in chosen:
        members = by_class[classes[ci]]
        idx += list(rng.choice(members, size=K, replace=len(members) < K))
    return np.array(idx, dtype=int)


def build_head(dim=512, n_classes=None, scale=16.0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, dim)
            with torch.no_grad():  # comeca como identidade = baseline sem treino
                self.proj.weight.copy_(torch.eye(dim))
                self.proj.bias.zero_()
            self.cls = nn.Parameter(torch.randn(n_classes, dim) * 0.01) if n_classes else None

        def embed(self, x):
            return F.normalize(self.proj(x), dim=1)

        def logits(self, e):
            return scale * F.linear(e, F.normalize(self.cls, dim=1))

    return Head()


def embed_numpy(head, emb):
    import torch
    head.eval()
    with torch.no_grad():
        out = head.embed(torch.from_numpy(np.asarray(emb, dtype=np.float32)))
    return out.numpy()


def batch_hard_triplet_loss(e, y, margin=0.3):
    """Para cada ancora: positivo MAIS DISTANTE e negativo MAIS PROXIMO do lote.
    Distancia = euclidiana ao quadrado entre vetores de norma 1 (= 2 - 2*cosseno)."""
    import torch
    d2 = (2.0 - 2.0 * (e @ e.T)).clamp(min=0.0)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool)
    pos = same & ~eye
    d_ap = (d2 * pos).max(dim=1).values
    d_an = (d2 + 1e6 * same.float()).min(dim=1).values
    return torch.relu(d_ap - d_an + margin).mean()


def train_head(emb_train, y_train, emb_val, val_pairs, loss_name, tr, seed=42, log=print):
    """Treina a camada e devolve a melhor epoca segundo o EER de pares da validacao."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    emb_train = np.asarray(emb_train, dtype=np.float32)
    classes = np.unique(y_train)
    remap = {int(c): i for i, c in enumerate(classes)}
    y_idx = np.array([remap[int(c)] for c in y_train])
    by_class = {int(i): np.flatnonzero(y_idx == i) for i in range(len(classes))}
    ce_cfg = tr.get("cross_entropy", {})
    head = build_head(emb_train.shape[1], len(classes) if loss_name == "cross_entropy" else None,
                      ce_cfg.get("scale", 16.0))
    opt = torch.optim.Adam(head.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    x = torch.from_numpy(emb_train)
    y = torch.from_numpy(y_idx)
    va, vb, vy = val_pairs

    def val_eer():
        e = embed_numpy(head, emb_val)
        return float(roc_eer(pair_scores(e, va, vb, "cosine"), vy)[0]["eer"])

    history = [{"epoch": 0, "train_loss": None, "val_eer": val_eer()}]
    best = {"epoch": 0, "val_eer": history[0]["val_eer"], "state": copy.deepcopy(head.state_dict())}
    log(f"  [{loss_name}] epoca 0 (baseline): val_EER={history[0]['val_eer']:.4f}")

    for epoch in range(1, tr["epochs"] + 1):
        head.train()
        losses = []
        if loss_name == "triplet":
            P, K = tr["triplet"]["P"], tr["triplet"]["K"]
            for _ in range(int(np.ceil(len(emb_train) / (P * K)))):
                idx = torch.from_numpy(pk_batch(by_class, P, K, rng))
                loss = batch_hard_triplet_loss(head.embed(x[idx]), y[idx], tr["triplet"]["margin"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        elif loss_name == "cross_entropy":
            bs = ce_cfg.get("batch_size", 128)
            perm = rng.permutation(len(emb_train))
            for i in range(0, len(perm), bs):
                idx = torch.from_numpy(perm[i:i + bs])
                loss = F.cross_entropy(head.logits(head.embed(x[idx])), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        else:
            raise ValueError(f"perda desconhecida: {loss_name}")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_eer": val_eer()}
        history.append(row)
        if row["val_eer"] < best["val_eer"]:   # so a validacao decide a melhor epoca
            best = {"epoch": epoch, "val_eer": row["val_eer"], "state": copy.deepcopy(head.state_dict())}
        if epoch % 5 == 0 or epoch == tr["epochs"]:
            log(f"  [{loss_name}] epoca {epoch}: loss={row['train_loss']:.4f} val_EER={row['val_eer']:.4f}")

    head.load_state_dict(best["state"])
    return {"head": head, "history": history, "best_epoch": best["epoch"], "best_val_eer": best["val_eer"]}
