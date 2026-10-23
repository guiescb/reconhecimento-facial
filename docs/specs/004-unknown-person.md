# 004 - Pessoa desconhecida

Status: rascunho (atualizar conforme a implementacao)

## Requisitos
- RF1: toda face abaixo do threshold e classificada como desconhecida (sem forcar escolha).
- RF2: o threshold vem da configuracao e foi escolhido na validacao (ADR-003).

## Criterios de aceitacao
- Face de pessoa nao cadastrada -> "desconhecida".
- Alterar o threshold na configuracao muda o resultado sem alterar codigo.
- O conjunto de teste nao e usado para escolher o threshold.
