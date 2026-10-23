import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

torch.set_grad_enabled(False)
print("torch", torch.__version__, "| numpy", np.__version__)

mtcnn = MTCNN(image_size=160, device="cpu")
model = InceptionResnetV1(pretrained="vggface2").eval()

img = Image.fromarray(np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8))
face = mtcnn(img)
print("detector rodou; face encontrada:", face is not None)

emb = model(torch.randn(1, 3, 160, 160))
print("embedding:", tuple(emb.shape))