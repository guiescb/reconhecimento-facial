# 001 - Deteccao facial

Status: implementada

## Requisitos
- RF1: dada uma imagem, detectar as faces e retornar caixas delimitadoras e pontos de referencia.
- RF2: alinhar e recortar cada face para o tamanho de entrada do modelo (pre-processamento): olhos na horizontal e recorte quadrado de 160x160 (lado = 1.3 x a caixa da face).
- RF3: reportar explicitamente os casos: nenhuma face, multiplas faces.
- RF4: no cadastro, rejeitar imagem inadequada: rosto pequeno, deteccao pouco confiavel ou imagem borrada (variancia do laplaciano). Os limites ficam em `configs/app.yaml`.
- RF5: arquivo que nao e imagem e rejeitado com erro tratado.

## Criterios de aceitacao
- Imagem com uma face -> 1 face alinhada retornada (160x160x3, uint8).
- Imagem sem face -> erro tratado (NoFaceDetected), sem excecao nao tratada.
- Imagem com varias faces -> todas retornadas, ou erro (MultipleFaces) quando o contexto exige uma so (cadastro).
- Rosto menor que `min_face_px`, confianca abaixo de `min_detection_prob` ou nitidez abaixo de `min_sharpness` -> InadequateImage com mensagem especifica.
- Apos o alinhamento, os olhos ficam na mesma linha.
- Testes automatizados cobrem todos os casos com imagens sinteticas e backend de mentira.

## Implementacao
`src/facerec/detection.py` (MTCNN via `MTCNNBackend`); testes em `tests/test_detection.py`.
