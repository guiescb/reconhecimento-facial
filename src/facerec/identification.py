"""Identificacao em conjunto aberto: pessoa reconhecida ou DESCONHECIDA (topico 32)."""
import numpy as np
from scipy.spatial.distance import cdist


def _score_matrix(probes, gallery, metric):
    if metric == "cosine":
        p = probes / np.linalg.norm(probes, axis=1, keepdims=True)
        g = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        return p @ g.T
    if metric == "euclidean":
        return -cdist(probes, gallery)
    raise ValueError(metric)


def open_set_identification(emb, labels, known_ids, unknown_ids, threshold,
                            n_enroll=3, seed=42, metric="cosine"):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    gallery, gallery_ids, probe_idx, probe_known = [], [], [], []
    for c in known_ids:
        idx = rng.permutation(np.flatnonzero(labels == c))
        enroll, rest = idx[:n_enroll], idx[n_enroll:]
        proto = emb[enroll].mean(axis=0)  # prototipo = media dos embeddings cadastrados
        gallery.append(proto / np.linalg.norm(proto))  # renormaliza (tamanho 1), como os demais
        gallery_ids.append(c)
        probe_idx += list(rest)
        probe_known += [True] * len(rest)
    for c in unknown_ids:
        idx = np.flatnonzero(labels == c)
        probe_idx += list(idx)
        probe_known += [False] * len(idx)

    probe_idx = np.array(probe_idx)
    known = np.array(probe_known)
    sims = _score_matrix(emb[probe_idx], np.stack(gallery), metric)
    best = sims.argmax(axis=1)
    accepted = sims.max(axis=1) >= threshold
    pred_id = np.array(gallery_ids)[best]
    true_id = labels[probe_idx]

    correct = known & accepted & (pred_id == true_id)
    rejected = known & ~accepted
    wrong_person = known & accepted & (pred_id != true_id)
    unknown_accepted = ~known & accepted
    unknown_rejected = ~known & ~accepted
    n_known, n_unknown = int(known.sum()), int((~known).sum())
    return {
        "n_probes_known": n_known,
        "n_probes_unknown": n_unknown,
        "correct_identification_rate": float(correct.sum() / max(n_known, 1)),
        "false_rejection_rate_known": float(rejected.sum() / max(n_known, 1)),
        "misidentification_rate_known": float(wrong_person.sum() / max(n_known, 1)),
        "false_acceptance_rate_unknown": float(unknown_accepted.sum() / max(n_unknown, 1)),
        "open_set_accuracy": float((correct.sum() + unknown_rejected.sum()) / len(known)),
    }
