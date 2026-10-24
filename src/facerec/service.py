"""Casos de uso da aplicacao: cadastro, reconhecimento e exclusao (specs 002, 003, 004 e 006).

Nao depende de interface nem de PyTorch: detector, embedder e banco sao recebidos de fora,
o que permite testar tudo com dubles (tests/fakes.py).

Regras principais:
- reconhecimento = galeria por PROTOTIPO + similaridade de COSSENO + threshold da configuracao
  (a melhor configuracao da Parte 1; ver ADR-001 e ADR-003);
- similaridade >= threshold -> reconhecida; senao -> "desconhecida" (spec 004);
- os logs registram so ids curtos e contagens: nunca nome nem dado biometrico (spec 006).
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from facerec.errors import (DuplicateEnrollment, FacerecError, InvalidName, InvalidPhotoCount)

log = logging.getLogger("facerec")


@dataclass
class Match:
    box: tuple
    recognized: bool
    score: Optional[float]       # melhor similaridade (None se nao ha ninguem cadastrado)
    person_id: Optional[str]     # None quando desconhecida
    name: Optional[str]


class FaceService:
    def __init__(self, detector, embedder, store, cfg):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        rec, enr = cfg["recognition"], cfg["enrollment"]
        self.threshold = float(rec["threshold"])
        self.duplicate_threshold = float(rec["duplicate_threshold"])
        self.min_photos = int(enr["min_photos"])
        self.max_photos = int(enr["max_photos"])
        self.store_photos = bool(enr["store_photos"])

    def has_people(self):
        return self.store.count_persons() > 0

    def list_people(self):
        return self.store.list_persons()

    def enroll(self, name, images):
        """Cadastra uma pessoa a partir de varias fotos. Qualquer problema levanta um erro
        tratado e NAO altera o banco."""
        name = (name or "").strip()
        if not name:
            raise InvalidName("informe o nome da pessoa.")
        if not self.min_photos <= len(images) <= self.max_photos:
            raise InvalidPhotoCount(
                f"envie entre {self.min_photos} e {self.max_photos} fotos (recebidas: {len(images)}).")
        faces = []
        for i, image in enumerate(images, 1):
            try:
                faces.append(self.detector.detect_single(image))
            except FacerecError as e:
                e.photo_index = i
                log.info("cadastro rejeitado: %s (foto %d)", type(e).__name__, i)
                raise
        vectors = self.embedder.embed(np.stack([f.crop for f in faces]))
        _, _, protos = self.store.prototypes()
        if len(protos):
            best = float((vectors @ protos.T).max())
            if best >= self.duplicate_threshold:
                log.info("cadastro rejeitado: DuplicateEnrollment (similaridade %.3f)", best)
                raise DuplicateEnrollment(
                    f"essa pessoa parece já estar cadastrada (similaridade {best:.2f}).")
        photos = [f.crop for f in faces] if self.store_photos else None
        try:
            person_id = self.store.add_person(name, vectors, photos)
        except FacerecError as e:
            log.info("cadastro rejeitado: %s", type(e).__name__)
            raise
        log.info("pessoa cadastrada: id=%s, %d embeddings", person_id[:8], len(vectors))
        return person_id

    def recognize(self, image):
        """Uma Match por face encontrada (lista vazia se nao houver face)."""
        faces = self.detector.detect(image)
        if not faces:
            return []
        ids, names, protos = self.store.prototypes()
        if not len(ids):
            return [Match(f.box, False, None, None, None) for f in faces]
        vectors = self.embedder.embed(np.stack([f.crop for f in faces]))
        sims = vectors @ protos.T
        matches = []
        for face, row in zip(faces, sims):
            k = int(row.argmax())
            score = float(row[k])
            ok = score >= self.threshold
            matches.append(Match(face.box, ok, score, ids[k] if ok else None,
                                 names[k] if ok else None))
        log.info("reconhecimento: %d face(s), %d reconhecida(s)",
                 len(matches), sum(m.recognized for m in matches))
        return matches

    def delete_person(self, person_id):
        self.store.delete_person(person_id)
        log.info("pessoa excluída: id=%s", person_id[:8])
