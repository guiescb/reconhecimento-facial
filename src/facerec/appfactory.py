"""Montagem da aplicacao a partir de configs/app.yaml (configuracao externa, spec 004)."""
import logging
from pathlib import Path


def setup_logging(cfg):
    """Log em arquivo (a pasta logs/ e ignorada pelo Git). Nunca registra nomes nem biometria."""
    lc = cfg["logging"]
    path = Path(lc["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("facerec")
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(lc.get("level", "INFO"))
    return logger


def build_service(cfg):
    """Cria detector (MTCNN), embedder (FaceNet), banco local e o servico da aplicacao."""
    from facerec.detection import FaceDetector, MTCNNBackend
    from facerec.embedder import FaceEmbedder
    from facerec.service import FaceService
    from facerec.store import PersonStore

    d, m, s = cfg["detection"], cfg["model"], cfg["storage"]
    detector = FaceDetector(MTCNNBackend(m["device"]), d["image_size"], d["crop_scale"],
                            d["min_detection_prob"], d["min_face_px"], d["min_sharpness"])
    embedder = FaceEmbedder(m["pretrained"], m["device"])
    store = PersonStore(s["db_path"], s["photos_dir"])
    return FaceService(detector, embedder, store, cfg)
