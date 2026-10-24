"""ADR-002 e spec 006: armazenamento local e exclusao completa."""
import sqlite3
 
import numpy as np
import pytest
 
from facerec.errors import PersonAlreadyExists, PersonNotFound
from facerec.store import PersonStore
 
from fakes import identity_vector
 
 
def vectors(ident, n=3):
    return np.stack([identity_vector(ident)] * n)
 
 
def count(db_path, table):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()
 
 
def test_banco_inicial_vazio(tmp_path):
    store = PersonStore(tmp_path / "t.db")
    assert store.count_persons() == 0 and store.list_persons() == []
    assert len(store.prototypes()[0]) == 0
 
 
def test_cadastro_cria_registro_com_identificador_e_embeddings(tmp_path):
    store = PersonStore(tmp_path / "t.db")
    pid = store.add_person("Ana", vectors(1))
    assert store.list_persons() == [{"id": pid, "name": "Ana", "n_embeddings": 3}]
    ids, names, protos = store.prototypes()
    assert ids == [pid] and names == ["Ana"]
    assert protos.shape == (1, 512) and abs(np.linalg.norm(protos[0]) - 1) < 1e-5
 
 
def test_nome_repetido_e_rejeitado_sem_diferenciar_maiusculas(tmp_path):
    store = PersonStore(tmp_path / "t.db")
    store.add_person("Ana", vectors(1))
    with pytest.raises(PersonAlreadyExists):
        store.add_person("ana", vectors(2))
    assert count(tmp_path / "t.db", "persons") == 1
    assert count(tmp_path / "t.db", "embeddings") == 3
 
 
def test_exclusao_remove_cadastro_e_embeddings_em_cascata(tmp_path):
    store = PersonStore(tmp_path / "t.db")
    keep = store.add_person("Ana", vectors(1))
    gone = store.add_person("Bia", vectors(2))
    store.delete_person(gone)
    assert [p["id"] for p in store.list_persons()] == [keep]
    assert count(tmp_path / "t.db", "persons") == 1
    assert count(tmp_path / "t.db", "embeddings") == 3   # so os da Ana: cascata funcionou
 
 
def test_exclusao_remove_as_fotos_armazenadas(tmp_path):
    store = PersonStore(tmp_path / "t.db", tmp_path / "fotos")
    photos = [np.zeros((160, 160, 3), dtype=np.uint8)] * 3
    pid = store.add_person("Ana", vectors(1), photos)
    assert len(list((tmp_path / "fotos" / pid).glob("*.jpg"))) == 3
    store.delete_person(pid)
    assert not (tmp_path / "fotos" / pid).exists()
 
 
def test_excluir_pessoa_inexistente_levanta_erro_tratado(tmp_path):
    with pytest.raises(PersonNotFound):
        PersonStore(tmp_path / "t.db").delete_person("nao-existe")
 
 
def test_falha_no_cadastro_nao_deixa_fotos_nem_linhas(tmp_path):
    store = PersonStore(tmp_path / "t.db", tmp_path / "fotos")
    photos = [np.zeros((160, 160, 3), dtype=np.uint8)] * 3
    first = store.add_person("Ana", vectors(1), photos)
    with pytest.raises(PersonAlreadyExists):
        store.add_person("Ana", vectors(2), photos)
    assert [p.name for p in (tmp_path / "fotos").iterdir()] == [first]
    assert count(tmp_path / "t.db", "persons") == 1