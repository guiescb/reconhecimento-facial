# 001 - Deteccao facial

Status: rascunho (atualizar conforme a implementacao)

## Requisitos
- RF1: dada uma imagem, detectar as faces e retornar caixas delimitadoras e pontos de referencia.
- RF2: alinhar e recortar cada face para o tamanho de entrada do modelo (pre-processamento).
- RF3: reportar explicitamente os casos: nenhuma face, multiplas faces.

## Criterios de aceitacao
- Imagem com uma face -> 1 face alinhada retornada.
- Imagem sem face -> erro tratado (NoFaceDetected), sem excecao nao tratada.
- Imagem com varias faces -> todas retornadas, ou erro (MultipleFaces) quando o contexto exige uma so (cadastro).
- Testes automatizados cobrem os tres casos com imagens sinteticas ou publicas.
