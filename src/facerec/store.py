"""Armazenamento local em SQLite (ADR-002).

Privacy by Design (specs 002 e 006):
- minimizacao: so guarda id, nome e embeddings (fotos apenas se store_photos estiver ligado);
- exclusao em cascata (PRAGMA foreign_keys = ON em TODA conexao; o SQLite nao liga isso sozinho);
- secure_delete: o conteudo apagado e sobrescrito com zeros no arquivo, nao so "esquecido".
"""
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from facerec.errors import PersonAlreadyExists, PersonNotFound

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
"""


class PersonStore:
    def __init__(self, db_path, photos_dir=None):
        self.db_path = Path(db_path)
        self.photos_dir = Path(photos_dir) if photos_dir else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA secure_delete = ON")
            with con:  # confirma (commit) no sucesso e desfaz (rollback) em caso de erro
                yield con
        finally:
            con.close()

    def count_persons(self):
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    def list_persons(self):
        with self._connect() as con:
            rows = con.execute(
                "SELECT p.id, p.name, COUNT(e.id) FROM persons p "
                "LEFT JOIN embeddings e ON e.person_id = p.id "
                "GROUP BY p.id ORDER BY p.name COLLATE NOCASE").fetchall()
        return [{"id": r[0], "name": r[1], "n_embeddings": r[2]} for r in rows]

    def add_person(self, name, vectors, photos=None):
        """Cadastra a pessoa e seus embeddings numa unica transacao. Devolve o id (uuid).
        Se algo falhar (nome repetido, erro ao salvar fotos), nada fica no banco nem em disco."""
        vectors = np.asarray(vectors, dtype=np.float32)
        person_id = uuid.uuid4().hex
        photo_dir = self.photos_dir / person_id if (photos is not None and self.photos_dir) else None
        try:
            with self._connect() as con:
                try:
                    con.execute("INSERT INTO persons (id, name, created_at) VALUES (?, ?, ?)",
                                (person_id, name, datetime.now(timezone.utc).isoformat()))
                except sqlite3.IntegrityError as e:
                    raise PersonAlreadyExists() from e
                con.executemany("INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                                [(person_id, v.tobytes()) for v in vectors])
                if photo_dir is not None:
                    photo_dir.mkdir(parents=True, exist_ok=True)
                    for i, photo in enumerate(photos, 1):
                        Image.fromarray(np.asarray(photo, dtype=np.uint8)).save(
                            photo_dir / f"{i:02d}.jpg", quality=92)
        except Exception:
            if photo_dir is not None:
                shutil.rmtree(photo_dir, ignore_errors=True)
            raise
        return person_id

    def prototypes(self):
        """(ids, nomes, matriz [n_pessoas, dim]): prototipo = media normalizada dos embeddings
        da pessoa, o mesmo metodo da galeria por prototipo do E02."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT p.id, p.name, e.vector FROM persons p JOIN embeddings e "
                "ON e.person_id = p.id ORDER BY p.created_at, p.id, e.id").fetchall()
        groups = {}
        for pid, name, blob in rows:
            groups.setdefault(pid, {"name": name, "vectors": []})["vectors"].append(
                np.frombuffer(blob, dtype=np.float32))
        if not groups:
            return [], [], np.zeros((0, 0), dtype=np.float32)
        ids = list(groups)
        names = [groups[i]["name"] for i in ids]
        protos = np.stack([np.mean(groups[i]["vectors"], axis=0) for i in ids]).astype(np.float32)
        protos /= np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12)
        return ids, names, protos

    def delete_person(self, person_id):
        """Remove cadastro, embeddings (cascata) e fotos armazenadas da pessoa."""
        with self._connect() as con:
            if con.execute("DELETE FROM persons WHERE id = ?", (person_id,)).rowcount == 0:
                raise PersonNotFound()
        if self.photos_dir:
            shutil.rmtree(self.photos_dir / person_id, ignore_errors=True)
