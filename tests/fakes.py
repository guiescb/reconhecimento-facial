"""Detector e embedder de mentira: testam a logica da aplicacao sem rede, sem PyTorch e sem
nenhuma foto real (dados sinteticos, como pede o enunciado, topico 23)."""
import numpy as np

from facerec.detection import Face
from facerec.errors import InadequateImage, MultipleFaces, NoFaceDetected

DIM = 512


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def identity_vector(ident):
    return _unit(np.random.default_rng(1000 + ident).normal(size=DIM))


class FakeImage:
    """Imagem sintetica: uma identidade por face e, opcionalmente, um problema de qualidade.
    `noise_seed` simula a variacao entre fotos da mesma pessoa."""

    def __init__(self, identities, quality_problem=None, noise_seed=0):
        self.identities = list(identities)
        self.quality_problem = quality_problem
        self.noise_seed = noise_seed


class FakeDetector:
    def detect(self, image):
        faces = []
        for k, ident in enumerate(image.identities):
            crop = np.zeros((160, 160, 3), dtype=np.uint8)
            crop[0, 0, 0] = ident
            crop[0, 0, 1] = image.noise_seed % 256
            faces.append(Face((10 + 200 * k, 10, 150 + 200 * k, 150), 0.99, crop, 100.0))
        return faces

    def detect_single(self, image):
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetected()
        if len(faces) > 1:
            raise MultipleFaces(len(faces))
        if image.quality_problem:
            raise InadequateImage(image.quality_problem)
        return faces[0]


class FakeEmbedder:
    def embed(self, crops):
        out = []
        for crop in crops:
            ident, noise = int(crop[0, 0, 0]), int(crop[0, 0, 1])
            v = identity_vector(ident)
            if noise:
                v = _unit(v + 0.3 * _unit(np.random.default_rng(noise).normal(size=DIM)))
            out.append(v)
        return np.stack(out)
