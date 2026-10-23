"""E02 - Conjunto aberto: tipo de galeria (prototipo x vizinho mais proximo) e threshold
calibrado no proprio conjunto aberto da VALIDACAO (topicos 28, 29 e 32).

O sorteio das fotos de cadastro e identico ao do E01 (mesma seed, mesma ordem), entao
"prototipo + threshold de EER por pares" reproduz exatamente os numeros do E01.
"""
import numpy as np

from facerec.metrics import threshold_at_far


def build_open_set(labels, known_ids, unknown_ids, n_enroll=3, seed=42):
    """Separa fotos de CADASTRO (galeria) das fotos de TESTE (probes).

    Pessoa conhecida: n_enroll fotos viram cadastro, as demais sao probes (devem ser reconhecidas).
    Pessoa desconhecida: todas as fotos sao probes (devem ser rejeitadas).
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    enroll_idx, enroll_ids, probe_idx, probe_known = [], [], [], []
    for c in known_ids:
        idx = rng.permutation(np.flatnonzero(labels == c))
        enroll, rest = idx[:n_enroll], idx[n_enroll:]
        enroll_idx += list(enroll)
        enroll_ids += [c] * len(enroll)
        probe_idx += list(rest)
        probe_known += [True] * len(rest)
    for c in unknown_ids:
        idx = np.flatnonzero(labels == c)
        probe_idx += list(idx)
        probe_known += [False] * len(idx)
    probe_idx = np.array(probe_idx, dtype=int)
    return {
        "enroll_idx": np.array(enroll_idx, dtype=int),
        "enroll_ids": np.array(enroll_ids),
        "probe_idx": probe_idx,
        "probe_known": np.array(probe_known, dtype=bool),
        "probe_true": labels[probe_idx],
    }


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def identity_scores(emb, os_split, method="prototype"):
    """Similaridade (cosseno) de cada probe com cada pessoa cadastrada.

    - "prototype": compara com a MEDIA normalizada das fotos de cadastro da pessoa.
    - "nearest":   compara com cada foto de cadastro e fica com a MAIS parecida (1-NN).
    Retorna (matriz [n_probes, n_pessoas], ids na ordem das colunas).
    """
    enroll_ids = os_split["enroll_ids"]
    ids = np.array(list(dict.fromkeys(enroll_ids.tolist())))
    probes = _unit(emb[os_split["probe_idx"]])
    gallery = emb[os_split["enroll_idx"]]
    if method == "prototype":
        protos = np.stack([gallery[enroll_ids == c].mean(axis=0) for c in ids])
        return probes @ _unit(protos).T, ids
    if method == "nearest":
        sims = probes @ _unit(gallery).T
        cols = [sims[:, enroll_ids == c].max(axis=1) for c in ids]
        return np.stack(cols, axis=1), ids
    raise ValueError(f"metodo de galeria desconhecido: {method}")


def evaluate_open_set(scores, ids, os_split, threshold):
    """Regra: melhor similaridade >= threshold -> reconhecida (como a pessoa mais parecida);
    senao -> desconhecida."""
    known, true = os_split["probe_known"], os_split["probe_true"]
    top = scores.max(axis=1)
    pred = ids[scores.argmax(axis=1)]
    accepted = top >= threshold
    correct = known & accepted & (pred == true)
    rejected = known & ~accepted
    wrong_person = known & accepted & (pred != true)
    unknown_accepted = ~known & accepted
    unknown_rejected = ~known & ~accepted
    n_known, n_unknown = int(known.sum()), int((~known).sum())
    return {
        "threshold": float(threshold),
        "n_probes_known": n_known,
        "n_probes_unknown": n_unknown,
        "correct_identification_rate": float(correct.sum() / max(n_known, 1)),
        "false_rejection_rate_known": float(rejected.sum() / max(n_known, 1)),
        "misidentification_rate_known": float(wrong_person.sum() / max(n_known, 1)),
        "false_acceptance_rate_unknown": float(unknown_accepted.sum() / max(n_unknown, 1)),
        "open_set_accuracy": float((correct.sum() + unknown_rejected.sum()) / len(known)),
        "n_unknown_identities": int(len(np.unique(true[~known]))),
        "unknown_identities_with_acceptance": int(len(np.unique(true[unknown_accepted]))),
    }


def open_set_curve(scores, ids, os_split, n_points=200):
    """Varre thresholds e devolve (thresholds, FAR dos desconhecidos, identificacao correta).
    So para REPORTAR/plotar; nunca para escolher threshold no teste."""
    top = scores.max(axis=1)
    thrs = np.linspace(top.min(), top.max(), n_points)
    far, cir = [], []
    for t in thrs:
        r = evaluate_open_set(scores, ids, os_split, t)
        far.append(r["false_acceptance_rate_unknown"])
        cir.append(r["correct_identification_rate"])
    return thrs, np.array(far), np.array(cir)


def collect_unknown_scores(emb, labels, val_ids, method, known_fraction=0.5, n_enroll=3,
                           n_repeats=20, seed=42):
    """Simula varias vezes a aplicacao SO com pessoas da validacao (metade cadastrada, metade
    desconhecida) e junta a melhor similaridade de cada foto desconhecida. Repetir o sorteio
    deixa a estimativa menos dependente de uma divisao conhecidos/desconhecidos so."""
    val_ids = np.asarray(val_ids)
    pooled = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + 1000 + r)
        perm = rng.permutation(val_ids)
        n_known = int(round(known_fraction * len(perm)))
        split = build_open_set(labels, perm[:n_known], perm[n_known:], n_enroll, seed + r)
        scores, _ = identity_scores(emb, split, method)
        pooled.append(scores.max(axis=1)[~split["probe_known"]])
    return np.concatenate(pooled)


def threshold_for_far(unknown_scores, far_target):
    """Menor threshold que deixa no maximo far_target dos desconhecidos (da validacao) aceitos."""
    return threshold_at_far(unknown_scores, np.zeros(len(unknown_scores), dtype=int), far_target)
