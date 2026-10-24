"""Deteccao, alinhamento e controle de qualidade de faces (spec 001).

Fluxo: backend (MTCNN) acha caixas, probabilidades e pontos de referencia ->
alinhamento (olhos na horizontal) -> recorte quadrado 160x160 -> controle de qualidade.
O backend e recebido de fora, entao os testes usam um backend de mentira (sem PyTorch).
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from facerec.errors import InadequateImage, MultipleFaces, NoFaceDetected


@dataclass
class Face:
    box: tuple          # (x1, y1, x2, y2) na imagem original
    prob: float         # confianca do detector
    crop: np.ndarray    # recorte alinhado uint8 [size, size, 3] (RGB)
    sharpness: float    # variancia do laplaciano do recorte


def load_image(file):
    """Abre um arquivo/stream como imagem RGB. Arquivo que nao e imagem -> InadequateImage."""
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        img = Image.open(file)
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise InadequateImage("o arquivo enviado não é uma imagem válida.") from e


def sharpness(rgb):
    """Nitidez = variancia do laplaciano da imagem em tons de cinza (0 para imagem lisa)."""
    gray = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    return float(lap.var())


def level_eyes(image, left_eye, right_eye, center):
    """Gira a imagem em torno de `center` para deixar a linha dos olhos na horizontal."""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    return image.rotate(angle, center=tuple(center), resample=Image.BILINEAR)


def align_crop(image, box, points=None, size=160, crop_scale=1.3):
    """Recorte quadrado (lado = crop_scale x maior lado da caixa) centrado na face, com os olhos
    na horizontal quando ha pontos de referencia, redimensionado para size x size."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if points is not None:
        pts = np.asarray(points, dtype=float)
        image = level_eyes(image, pts[0], pts[1], (cx, cy))
    half = crop_scale * max(x2 - x1, y2 - y1) / 2
    crop = image.crop((int(round(cx - half)), int(round(cy - half)),
                       int(round(cx + half)), int(round(cy + half))))
    crop = crop.resize((size, size), Image.BILINEAR)
    return np.asarray(crop.convert("RGB"), dtype=np.uint8)


class MTCNNBackend:
    """Detector real (facenet-pytorch). O PyTorch so e importado quando a classe e usada."""

    def __init__(self, device="cpu"):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image):
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        if boxes is None:
            return [], [], []
        return list(boxes), [float(p) for p in probs], list(points)


class FaceDetector:
    def __init__(self, backend, image_size=160, crop_scale=1.3, min_prob=0.9,
                 min_face_px=60, min_sharpness=25.0):
        self.backend = backend
        self.image_size = int(image_size)
        self.crop_scale = float(crop_scale)
        self.min_prob = float(min_prob)
        self.min_face_px = float(min_face_px)
        self.min_sharpness = float(min_sharpness)

    def detect(self, image):
        """Todas as faces encontradas (sem controle de qualidade), usada no reconhecimento."""
        boxes, probs, points = self.backend.detect(image)
        faces = []
        for box, prob, pts in zip(boxes, probs, points):
            box = tuple(float(v) for v in box)
            crop = align_crop(image, box, pts, self.image_size, self.crop_scale)
            faces.append(Face(box, float(prob), crop, sharpness(crop)))
        return faces

    def detect_single(self, image):
        """Usada no cadastro: exige exatamente uma face e de qualidade adequada."""
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetected()
        if len(faces) > 1:
            raise MultipleFaces(len(faces))
        face = faces[0]
        x1, y1, x2, y2 = face.box
        if min(x2 - x1, y2 - y1) < self.min_face_px:
            raise InadequateImage("rosto muito pequeno; aproxime-se da câmera.")
        if face.prob < self.min_prob:
            raise InadequateImage("detecção pouco confiável; melhore a iluminação e olhe de frente.")
        if face.sharpness < self.min_sharpness:
            raise InadequateImage("imagem borrada; tire a foto novamente com a câmera firme.")
        return face
