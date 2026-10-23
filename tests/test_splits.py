import numpy as np
import pytest

from facerec.splits import check_no_leakage, split_identities


def _labels():
    return np.repeat(np.arange(20), 5)


def test_identidades_disjuntas():
    labels = _labels()
    s = split_identities(labels, (0.6, 0.2, 0.2), seed=1)
    assert check_no_leakage(s, labels)
    assert sum(len(v) for v in s.values()) == len(labels)


def test_split_e_reprodutivel():
    labels = _labels()
    a = split_identities(labels, seed=7)
    b = split_identities(labels, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_detecta_vazamento_de_identidade():
    labels = _labels()
    s = split_identities(labels, seed=1)
    s["test"] = np.concatenate([s["test"], s["train"][:1]])
    with pytest.raises(AssertionError):
        check_no_leakage(s, labels)


def test_detecta_imagem_duplicada_entre_conjuntos():
    labels = np.array([0, 0, 1, 1])
    images = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    images[0] += 1
    images[1] += 2
    images[2] += 3
    images[3] += 3  # imagem 3 identica a imagem 2
    s = {"train": np.array([0, 1, 2]), "test": np.array([3])}
    s_ok = {"train": np.array([0, 1]), "test": np.array([2, 3])}
    assert check_no_leakage(s_ok, labels, images)
    with pytest.raises(AssertionError):
        check_no_leakage({"train": np.array([0, 2]), "test": np.array([1, 3])}, np.array([0, 1, 2, 3]), images)
