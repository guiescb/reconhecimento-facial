# 003 - Reconhecimento facial

Status: rascunho (atualizar conforme a implementacao)

## Requisitos
- RF1: receber imagem, video ou camera; Deteccao -> Embedding -> Comparacao -> Threshold -> Reconhecido/desconhecido.
- RF2: usar a melhor configuracao da Parte 1 (ou justificar a diferenca em ADR).
- RF3: retornar identificador e pontuacao de similaridade.

## Criterios de aceitacao
- Pessoa cadastrada com foto nova -> reconhecida com similaridade >= threshold.
- A avaliacao reporta accuracy, F1, FAR, FRR e EER.
- Mesma entrada e mesma configuracao produzem o mesmo resultado.
