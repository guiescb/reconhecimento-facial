# ADR-001 - Modelo de reconhecimento facial

- Status: aceita
- Data: 2026-10-23

## Contexto
Hardware: apenas CPU (Intel HD Graphics 520, 8 GB de RAM). O modelo precisa gerar embeddings em
tempo aceitavel e permitir alguma forma de fine-tuning parcial.

## Candidatos
FaceNet (InceptionResnetV1 pre-treinado), MobileFaceNet, ResNet pequena.

## Decisao
FaceNet (InceptionResnetV1, pesos VGGFace2, embeddings de 512 dimensoes normalizados), sem treino
adicional, com galeria por prototipo, similaridade de cosseno e threshold definido no ADR-003.

Motivos, com base nos experimentos:
- E01: o modelo pre-treinado ja separa bem as pessoas do LFW (AUC 0.9995, EER 0.68% no teste).
- E02: o ponto fraco era a identificacao de desconhecidos, resolvido com a calibracao do threshold
  na validacao, sem mudar o modelo.
- E03: treinar uma camada de projecao sobre os embeddings congelados (Triplet e Cross-Entropy)
  nao melhorou nada; a melhor epoca foi sempre o baseline.
- Custo: extrair os embeddings de 4324 imagens levou 657 s em CPU, uma unica vez (com cache).

## Consequencias
- MobileFaceNet e ResNet pequena eram candidatas, mas nao foram avaliadas neste trabalho. A escolha
  se apoia nos resultados obtidos com o FaceNet, no hardware e no prazo, e nao em uma comparacao
  entre modelos.
- O FaceNet foi pre-treinado no VGGFace2, que pode ter identidades em comum com o LFW, entao os
  resultados podem estar otimistas.
- O que foi treinado no E03 foi so uma camada de projecao; as camadas internas do FaceNet nao
  foram ajustadas.
- A aplicacao da Parte 2 usa este mesmo modelo. Se usar outro, a justificativa deve ser registrada
  aqui.

## Relacionado
ADR-003 (threshold) e experimentos E01, E02 e E03 (ver README).