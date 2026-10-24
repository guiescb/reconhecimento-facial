"""Embeddings do FaceNet para a aplicacao (modelo carregado uma unica vez).

Usa a MESMA padronizacao e normalizacao de facerec.embeddings.extract_embeddings (Parte 1):
(x - 127.5) / 128 e depois normalizacao L2. Assim o threshold calibrado na Parte 1 continua valido.
"""
import numpy as np


class FaceEmbedder:
    def __init__(self, pretrained="vggface2", device="cpu"):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._torch = torch
        self.device = device
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)

    def embed(self, faces_uint8):
        """faces_uint8: array [n, 160, 160, 3] (RGB) -> embeddings float32 [n, 512], norma 1."""
        torch = self._torch
        x = torch.from_numpy(np.asarray(faces_uint8, dtype=np.uint8)).permute(0, 3, 1, 2).float()
        x = (x - 127.5) / 128.0
        with torch.no_grad():
            e = torch.nn.functional.normalize(self.model(x.to(self.device)), dim=1)
        return e.cpu().numpy().astype(np.float32)
