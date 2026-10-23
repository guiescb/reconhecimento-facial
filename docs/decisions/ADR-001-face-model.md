# ADR-001 - Modelo de reconhecimento facial

- Status: proposta (confirmar apos os experimentos da Parte 1)
- Data: 2026-10-23

## Contexto
Hardware: apenas CPU (Intel HD Graphics 520). O modelo precisa gerar embeddings em tempo aceitavel e permitir fine-tuning parcial.

## Candidatos
FaceNet (InceptionResnetV1 pre-treinado), MobileFaceNet, ResNet pequena. Comparacao via experimentos E01.. com a mesma divisao de dados.

## Decisao
A definir com base em metricas (EER, F1), custo computacional e tempo de inferencia em CPU. Se a aplicacao usar modelo diferente do melhor da Parte 1, justificar aqui.
