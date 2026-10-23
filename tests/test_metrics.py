import numpy as np

from facerec.data import synthetic_embeddings
from facerec.identification import open_set_identification
from facerec.metrics import evaluate_at_threshold, make_pairs, pair_scores, roc_eer, threshold_at_far


def test_pares_balanceados_e_corretos():
    labels = np.repeat(np.arange(10), 4)
    a, b, y = make_pairs(labels, 200, seed=3)
    assert y.sum() == 100
    assert np.all(labels[a[y == 1]] == labels[b[y == 1]])
    assert np.all(labels[a[y == 0]] != labels[b[y == 0]])


def test_eer_perfeito_e_aleatorio():
    y = np.array([1] * 50 + [0] * 50)
    perfeito = np.array([0.9] * 50 + [0.1] * 50)
    s, _ = roc_eer(perfeito, y)
    assert s["eer"] == 0.0 and s["auc"] == 1.0
    rng = np.random.default_rng(0)
    s2, _ = roc_eer(rng.random(2000), np.array([1, 0] * 1000))
    assert 0.4 < s2["eer"] < 0.6


def test_far_frr_na_matriz_de_confusao():
    scores = np.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.1])
    y = np.array([1, 1, 1, 0, 0, 0])
    r = evaluate_at_threshold(scores, y, 0.5)
    assert r["confusion_matrix"] == [[2, 1], [1, 2]]
    assert abs(r["far"] - 1 / 3) < 1e-9 and abs(r["frr"] - 1 / 3) < 1e-9


def test_threshold_respeita_far_alvo():
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.normal(1, 0.3, 1000), rng.normal(0, 0.3, 1000)])
    y = np.array([1] * 1000 + [0] * 1000)
    thr = threshold_at_far(s, y, 0.05)
    assert evaluate_at_threshold(s, y, thr)["far"] <= 0.05


def test_similaridade_cosseno_e_euclidiana():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    c = pair_scores(emb, np.array([0, 0]), np.array([1, 2]), "cosine")
    e = pair_scores(emb, np.array([0, 0]), np.array([1, 2]), "euclidean")
    assert np.allclose(c, [1.0, 0.0])
    assert e[0] > e[1]  # mais parecido => score maior


def test_conjunto_aberto_com_dados_sinteticos():
    emb, labels = synthetic_embeddings(n_ids=30, seed=0)
    ids = np.unique(labels)
    out = open_set_identification(emb, labels, ids[:15], ids[15:], threshold=0.3, n_enroll=3)
    assert 0.0 <= out["correct_identification_rate"] <= 1.0
    assert out["n_probes_known"] > 0 and out["n_probes_unknown"] > 0
    # threshold impossivel: ninguem e aceito, todo desconhecido e corretamente rejeitado
    out2 = open_set_identification(emb, labels, ids[:15], ids[15:], threshold=2.0)
    assert out2["false_acceptance_rate_unknown"] == 0.0
    assert out2["correct_identification_rate"] == 0.0
