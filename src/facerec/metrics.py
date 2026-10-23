"""Pares, similaridade, ROC/EER, FAR/FRR e metricas no threshold (topicos 31 e 32)."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def make_pairs(labels, n_pairs, seed=42):
    """Pares balanceados: metade da MESMA pessoa (y=1) e metade de pessoas DIFERENTES (y=0)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    eligible = classes[counts >= 2]
    by_class = {c: np.flatnonzero(labels == c) for c in classes}
    n_gen = n_pairs // 2
    a, b, y = [], [], []
    for _ in range(n_gen):
        c = rng.choice(eligible)
        i, j = rng.choice(by_class[c], 2, replace=False)
        a.append(i), b.append(j), y.append(1)
    for _ in range(n_pairs - n_gen):
        c1, c2 = rng.choice(classes, 2, replace=False)
        a.append(rng.choice(by_class[c1])), b.append(rng.choice(by_class[c2])), y.append(0)
    return np.array(a), np.array(b), np.array(y)


def pair_scores(emb, a, b, metric="cosine"):
    """Quanto MAIOR o score, mais parecidas as faces (para euclidiana usamos -distancia)."""
    ea, eb = emb[a], emb[b]
    if metric == "cosine":
        na = np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1)
        return np.sum(ea * eb, axis=1) / np.maximum(na, 1e-12)
    if metric == "euclidean":
        return -np.linalg.norm(ea - eb, axis=1)
    raise ValueError(f"metrica desconhecida: {metric}")


def roc_eer(scores, y):
    fpr, tpr, thr = roc_curve(y, scores)
    frr = 1 - tpr
    k = int(np.argmin(np.abs(fpr - frr)))
    summary = {
        "auc": float(roc_auc_score(y, scores)),
        "eer": float((fpr[k] + frr[k]) / 2),
        "eer_threshold": float(thr[k]),
    }
    return summary, (fpr, tpr)


def threshold_at_far(scores, y, far_target=0.01):
    """Menor limiar cujo FAR (impostores aceitos) fica <= far_target."""
    impostor = np.sort(scores[y == 0])
    k = int(np.ceil((1 - far_target) * len(impostor)))
    k = min(max(k, 0), len(impostor) - 1)
    return float(np.nextafter(impostor[k], np.inf))


def evaluate_at_threshold(scores, y, threshold):
    """Regra do enunciado: score >= threshold -> mesma pessoa (reconhecida)."""
    y = np.asarray(y)
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
