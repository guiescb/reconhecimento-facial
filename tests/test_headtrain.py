import numpy as np
import pytest

from facerec.data import synthetic_embeddings
from facerec.headtrain import pk_batch
from facerec.metrics import make_pairs
from facerec.splits import split_identities


def test_pk_batch_tem_p_identidades_e_k_fotos_cada():
    labels = np.repeat(np.arange(10), 6)
    by_class = {c: np.flatnonzero(labels == c) for c in range(10)}
    idx = pk_batch(by_class, P=4, K=3, rng=np.random.default_rng(0))
    assert len(idx) == 12
    ids, counts = np.unique(labels[idx], return_counts=True)
    assert len(ids) == 4 and np.all(counts == 3)


torch = pytest.importorskip("torch")
from facerec.headtrain import batch_hard_triplet_loss, build_head, embed_numpy, train_head  # noqa: E402


def test_camada_comeca_como_identidade():
    emb, _ = synthetic_embeddings(n_ids=5, seed=1)
    out = embed_numpy(build_head(512), emb[:10])
    assert np.allclose(out, emb[:10], atol=1e-5)


def test_triplet_zero_quando_bem_separado_e_positiva_quando_misturado():
    f = torch.nn.functional.normalize
    y = torch.tensor([0, 0, 1, 1])
    bom = f(torch.tensor([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]]), dim=1)
    ruim = f(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), dim=1)
    assert batch_hard_triplet_loss(bom, y, 0.3).item() == pytest.approx(0.0, abs=1e-6)
    assert batch_hard_triplet_loss(ruim, y, 0.3).item() > 0.3


@pytest.mark.parametrize("loss_name", ["triplet", "cross_entropy"])
def test_treino_roda_e_melhor_epoca_nunca_pior_que_baseline_na_validacao(loss_name):
    emb, labels = synthetic_embeddings(n_ids=40, seed=3)
    sp = split_identities(labels, (0.6, 0.2, 0.2), seed=3)
    a, b, y = make_pairs(labels[sp["val"]], 400, seed=3)
    tr = {"epochs": 3, "lr": 1e-3, "weight_decay": 1e-4,
          "triplet": {"margin": 0.3, "P": 8, "K": 4}, "cross_entropy": {"batch_size": 32, "scale": 16.0}}
    out = train_head(emb[sp["train"]], labels[sp["train"]], emb[sp["val"]], (a, b, y),
                     loss_name, tr, seed=3, log=lambda *_: None)
    h = out["history"]
    assert len(h) == 4 and h[0]["epoch"] == 0
    assert out["best_val_eer"] <= h[0]["val_eer"] + 1e-12
    assert all(np.isfinite(r["train_loss"]) for r in h[1:])
