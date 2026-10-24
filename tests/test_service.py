"""Specs 002, 003, 004 e 006: cadastro, reconhecimento, pessoa desconhecida e exclusao."""
import logging
import sqlite3

import pytest

from facerec.errors import (DuplicateEnrollment, InadequateImage, InvalidPhotoCount,
                            MultipleFaces, NoFaceDetected, PersonAlreadyExists, PersonNotFound)
from facerec.service import FaceService
from facerec.store import PersonStore

from fakes import FakeDetector, FakeEmbedder, FakeImage


def make_cfg(threshold=0.716, store_photos=False):
    return {"recognition": {"threshold": threshold, "duplicate_threshold": 0.716},
            "enrollment": {"min_photos": 3, "max_photos": 10, "store_photos": store_photos}}


def make_service(tmp_path, threshold=0.716, store_photos=False):
    store = PersonStore(tmp_path / "t.db", tmp_path / "fotos")
    return FaceService(FakeDetector(), FakeEmbedder(), store, make_cfg(threshold, store_photos))


def photos(ident, n=3, start=1):
    return [FakeImage([ident], noise_seed=start + i) for i in range(n)]


def test_banco_inicial_vazio_nao_reconhece_ninguem(tmp_path):
    svc = make_service(tmp_path)
    assert not svc.has_people()
    (m,) = svc.recognize(FakeImage([1]))
    assert not m.recognized and m.score is None and m.name is None


def test_cadastro_valido_cria_registro_com_identificador_e_embeddings(tmp_path):
    svc = make_service(tmp_path)
    pid = svc.enroll("Ana", photos(1))
    assert svc.list_people() == [{"id": pid, "name": "Ana", "n_embeddings": 3}]


def test_cadastro_exige_o_numero_minimo_de_fotos(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(InvalidPhotoCount):
        svc.enroll("Ana", photos(1, n=2))
    assert not svc.has_people()


def test_cada_rejeicao_tem_erro_tratado_e_nao_altera_o_banco(tmp_path):
    svc = make_service(tmp_path)
    casos = [(FakeImage([]), NoFaceDetected),
             (FakeImage([1, 2]), MultipleFaces),
             (FakeImage([1], quality_problem="imagem borrada"), InadequateImage)]
    for ruim, erro in casos:
        with pytest.raises(erro) as info:
            svc.enroll("Ana", photos(1, n=2) + [ruim])
        assert info.value.photo_index == 3 and str(info.value).startswith("Foto 3:")
        assert not svc.has_people()


def test_segundo_cadastro_da_mesma_pessoa_e_detectado_como_duplicado(tmp_path):
    svc = make_service(tmp_path)
    svc.enroll("Ana", photos(1))
    with pytest.raises(DuplicateEnrollment):
        svc.enroll("Outro nome", photos(1, start=50))
    assert len(svc.list_people()) == 1


def test_nome_repetido_e_rejeitado(tmp_path):
    svc = make_service(tmp_path)
    svc.enroll("Ana", photos(1))
    with pytest.raises(PersonAlreadyExists):
        svc.enroll("ana", photos(2))
    assert len(svc.list_people()) == 1


def test_pessoa_cadastrada_com_foto_nova_e_reconhecida(tmp_path):
    svc = make_service(tmp_path)
    pid = svc.enroll("Ana", photos(1))
    (m,) = svc.recognize(FakeImage([1], noise_seed=77))
    assert m.recognized and m.name == "Ana" and m.person_id == pid and m.score >= 0.716


def test_pessoa_nao_cadastrada_e_desconhecida(tmp_path):
    svc = make_service(tmp_path)
    svc.enroll("Ana", photos(1))
    (m,) = svc.recognize(FakeImage([2]))
    assert not m.recognized and m.name is None and m.person_id is None and m.score < 0.716


def test_varias_faces_no_reconhecimento_sao_tratadas_uma_a_uma(tmp_path):
    svc = make_service(tmp_path)
    svc.enroll("Ana", photos(1))
    primeira, segunda = svc.recognize(FakeImage([1, 2], noise_seed=9))
    assert primeira.recognized and not segunda.recognized


def test_imagem_sem_face_no_reconhecimento_retorna_lista_vazia(tmp_path):
    assert make_service(tmp_path).recognize(FakeImage([])) == []


def test_mudar_o_threshold_na_configuracao_muda_o_resultado_sem_mudar_codigo(tmp_path):
    base = make_service(tmp_path)
    base.enroll("Ana", photos(1))
    rigido = FaceService(FakeDetector(), FakeEmbedder(), base.store, make_cfg(threshold=0.999))
    foto = FakeImage([1], noise_seed=77)
    assert base.recognize(foto)[0].recognized
    assert not rigido.recognize(foto)[0].recognized


def test_mesma_entrada_e_mesma_configuracao_dao_o_mesmo_resultado(tmp_path):
    svc = make_service(tmp_path)
    svc.enroll("Ana", photos(1))
    foto = FakeImage([1], noise_seed=5)
    assert svc.recognize(foto) == svc.recognize(foto)


def test_exclusao_remove_tudo_e_a_pessoa_deixa_de_ser_reconhecida(tmp_path):
    svc = make_service(tmp_path, store_photos=True)
    pid = svc.enroll("Ana", photos(1))
    assert (tmp_path / "fotos" / pid).exists()
    svc.delete_person(pid)
    assert not svc.has_people() and not (tmp_path / "fotos" / pid).exists()
    con = sqlite3.connect(tmp_path / "t.db")
    try:
        assert con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    finally:
        con.close()
    assert not svc.recognize(FakeImage([1], noise_seed=77))[0].recognized


def test_excluir_pessoa_inexistente_retorna_erro_tratado(tmp_path):
    with pytest.raises(PersonNotFound):
        make_service(tmp_path).delete_person("nao-existe")


def test_logs_nao_guardam_nome_nem_dados_biometricos(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="facerec")
    svc = make_service(tmp_path)
    pid = svc.enroll("Maria Silva", photos(1))
    svc.recognize(FakeImage([1], noise_seed=77))
    with pytest.raises(DuplicateEnrollment):
        svc.enroll("Joana Souza", photos(1, start=60))
    svc.delete_person(pid)
    texto = " ".join(r.getMessage() for r in caplog.records)
    assert texto  # os eventos foram registrados
    for proibido in ("Maria", "Silva", "Joana", "Souza"):
        assert proibido not in texto
