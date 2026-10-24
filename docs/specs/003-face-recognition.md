# 003 - Reconhecimento facial

Status: implementada

## Requisitos
- RF1: receber imagem ou camera; Deteccao -> Embedding -> Comparacao -> Threshold -> Reconhecido/desconhecido. (Video continuo nao foi implementado: a camera funciona por fotos.)
- RF2: usar a melhor configuracao da Parte 1: FaceNet (VGGFace2) sem treino extra, galeria por prototipo, similaridade de cosseno (ADR-001 e ADR-003).
- RF3: retornar identificador e pontuacao de similaridade, uma resposta por face encontrada.

## Criterios de aceitacao
- Pessoa cadastrada com foto nova -> reconhecida com similaridade >= threshold.
- Imagem com varias faces -> um resultado por face; imagem sem face -> lista vazia ("Nenhuma face detectada").
- A avaliacao (Parte 1) reporta accuracy, F1, FAR, FRR e EER (`results/`).
- Mesma entrada e mesma configuracao produzem o mesmo resultado.

## Implementacao
`src/facerec/service.py` (`recognize`), `src/facerec/embedder.py` (mesma padronizacao da Parte 1); testes em `tests/test_service.py`.
