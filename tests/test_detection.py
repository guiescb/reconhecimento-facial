"""Spec 001: deteccao, alinhamento e controle de qualidade (com backend de mentira)."""
import io
 
import numpy as np
import pytest
from PIL import Image
 
from facerec.detection import FaceDetector, align_crop, level_eyes, load_image, sharpness
from facerec.errors import InadequateImage, MultipleFaces, NoFaceDetected
 
 
class FakeBackend:
    def __init__(self, boxes=(), probs=(), points=None):
        self.boxes, self.probs = list(boxes), list(probs)
        self.points = points if points is not None else [None] * len(self.boxes)
 
    def detect(self, image):
        return self.boxes, self.probs, self.points
 
 
def noisy_image(size=400, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
 
 
def flat_image(size=400):
    return Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))
 
 
GOOD_BOX = (100, 100, 260, 280)
 
 
def test_nitidez_distingue_imagem_nitida_de_borrada():
    assert sharpness(np.asarray(noisy_image(160))) > 1000
    assert sharpness(np.asarray(flat_image(160))) == 0
 
 
def test_imagem_com_uma_face_retorna_face_alinhada_160x160():
    det = FaceDetector(FakeBackend([GOOD_BOX], [0.99]))
    face = det.detect_single(noisy_image())
    assert face.crop.shape == (160, 160, 3) and face.crop.dtype == np.uint8
 
 
def test_imagem_sem_face_levanta_erro_tratado():
    det = FaceDetector(FakeBackend())
    assert det.detect(noisy_image()) == []
    with pytest.raises(NoFaceDetected):
        det.detect_single(noisy_image())
 
 
def test_varias_faces_levanta_erro_no_cadastro_mas_detect_retorna_todas():
    det = FaceDetector(FakeBackend([GOOD_BOX, (10, 10, 100, 100)], [0.99, 0.95]))
    assert len(det.detect(noisy_image())) == 2
    with pytest.raises(MultipleFaces) as info:
        det.detect_single(noisy_image())
    assert info.value.n == 2
 
 
def test_rosto_pequeno_e_inadequado():
    det = FaceDetector(FakeBackend([(100, 100, 140, 140)], [0.99]), min_face_px=60)
    with pytest.raises(InadequateImage, match="pequeno"):
        det.detect_single(noisy_image())
 
 
def test_deteccao_pouco_confiavel_e_inadequada():
    det = FaceDetector(FakeBackend([GOOD_BOX], [0.5]), min_prob=0.9)
    with pytest.raises(InadequateImage, match="confiável"):
        det.detect_single(noisy_image())
 
 
def test_imagem_borrada_e_inadequada():
    det = FaceDetector(FakeBackend([GOOD_BOX], [0.99]), min_sharpness=25.0)
    with pytest.raises(InadequateImage, match="borrada"):
        det.detect_single(flat_image())
 
 
def _centroid_row(img):
    ys, _ = np.nonzero(np.asarray(img.convert("L")) > 200)
    return ys.mean()
 
 
def test_alinhamento_deixa_os_olhos_na_horizontal():
    left, right, center = (150, 180), (250, 200), (200, 190)   # olho direito mais baixo
    rows = []
    for eye in (left, right):
        arr = np.zeros((400, 400, 3), dtype=np.uint8)
        arr[eye[1] - 2:eye[1] + 3, eye[0] - 2:eye[0] + 3] = 255
        rows.append(_centroid_row(level_eyes(Image.fromarray(arr), left, right, center)))
    assert abs(rows[0] - rows[1]) < 1.0
    assert abs(rows[0] - 190) < 3   # girou em torno do centro, sem deslocar a face
 
 
def test_recorte_com_pontos_de_referencia_tem_tamanho_fixo():
    pts = np.array([[150, 180], [250, 200], [200, 230], [160, 260], [240, 260]], dtype=float)
    assert align_crop(noisy_image(), GOOD_BOX, pts).shape == (160, 160, 3)
 
 
def test_arquivo_que_nao_e_imagem_e_rejeitado():
    with pytest.raises(InadequateImage):
        load_image(io.BytesIO(b"isto nao e uma imagem"))