import numpy as np

from facerec.data import synthetic_embeddings
from facerec.identification import open_set_identification
from facerec.openset import (build_open_set, collect_unknown_scores, evaluate_open_set,
                             identity_scores, threshold_for_far)


def _data():
    emb, labels = synthetic_embeddings(n_ids=30, seed=0)
    ids = np.unique(labels)
    return emb, labels, ids[:15], ids[15:]


def test_cadastro_e_probes_nao_se_misturam():
    emb, labels, known, unknown = _data()
    s = build_open_set(labels, known, unknown, n_enroll=3)
    assert not set(s["enroll_idx"]) & set(s["probe_idx"])           # nenhuma foto nos dois lados
    assert all(np.sum(s["enroll_ids"] == c) == 3 for c in known)     # 3 fotos por conhecido
    assert not set(s["enroll_ids"]) & set(unknown)                   # desconhecidos nao cadastrados
    assert s["probe_known"].sum() > 0 and (~s["probe_known"]).sum() > 0


def test_prototipo_reproduz_o_e01():
    emb, labels, known, unknown = _data()
    s = build_open_set(labels, known, unknown, n_enroll=3, seed=42)
    sc, ids = identity_scores(emb, s, "prototype")
    novo = evaluate_open_set(sc, ids, s, 0.3)
    velho = open_set_identification(emb, labels, known, unknown, 0.3, n_enroll=3, seed=42)
    for k in ("correct_identification_rate", "false_rejection_rate_known", "misidentification_rate_known",
              "false_acceptance_rate_unknown", "open_set_accuracy"):
        assert np.isclose(novo[k], velho[k]), k


def test_vizinho_mais_proximo_acha_foto_cadastrada():
    emb, labels, known, unknown = _data()
    s = build_open_set(labels, known, unknown, n_enroll=3)
    # um probe identico a uma foto de cadastro tem similaridade 1 com ela no metodo "nearest"
    s2 = dict(s)
    s2["probe_idx"] = s["enroll_idx"][:1]
    s2["probe_known"] = np.array([True])
    s2["probe_true"] = labels[s["enroll_idx"][:1]]
    sc, ids = identity_scores(emb, s2, "nearest")
    assert np.isclose(sc.max(), 1.0, atol=1e-5)
    assert ids[sc.argmax()] == s2["probe_true"][0]


def test_threshold_impossivel_rejeita_todos():
    emb, labels, known, unknown = _data()
    s = build_open_set(labels, known, unknown)
    for method in ("prototype", "nearest"):
        sc, ids = identity_scores(emb, s, method)
        r = evaluate_open_set(sc, ids, s, 2.0)
        assert r["false_acceptance_rate_unknown"] == 0.0 and r["correct_identification_rate"] == 0.0


def test_threshold_calibrado_respeita_alvo_e_e_monotono():
    emb, labels, known, unknown = _data()
    pooled = collect_unknown_scores(emb, labels, np.unique(labels), "prototype", n_repeats=5)
    t5, t1 = threshold_for_far(pooled, 0.05), threshold_for_far(pooled, 0.01)
    assert t1 >= t5                                   # alvo mais rigido => threshold maior ou igual
    assert np.mean(pooled >= t5) <= 0.05              # FAR da validacao dentro do alvo
    assert np.mean(pooled >= t1) <= 0.01


def test_metodo_invalido():
    emb, labels, known, unknown = _data()
    s = build_open_set(labels, known, unknown)
    try:
        identity_scores(emb, s, "xyz")
    except ValueError:
        return
    raise AssertionError("deveria levantar ValueError")
