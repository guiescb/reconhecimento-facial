# 004 - Pessoa desconhecida

Status: implementada

## Requisitos
- RF1: toda face abaixo do threshold e classificada como desconhecida (sem forcar escolha).
- RF2: o threshold vem da configuracao (`configs/app.yaml`) e foi escolhido na validacao (ADR-003).
- RF3: sem ninguem cadastrado, toda face e desconhecida e a interface avisa.

## Criterios de aceitacao
- Face de pessoa nao cadastrada -> "desconhecida" (sem nome nem id), com similaridade abaixo do threshold.
- Banco vazio -> face detectada vira desconhecida, sem pontuacao.
- Alterar o threshold na configuracao muda o resultado sem alterar codigo.
- O conjunto de teste nao e usado para escolher o threshold.
- A interface mostra a melhor similaridade tambem para desconhecidos, para apoiar a recalibracao com camera real.

## Implementacao
`src/facerec/service.py`; testes em `tests/test_service.py`.
