# ADR-003 - Limiar de reconhecimento

- Status: proposta (valor definido na validacao)
- Data: 2026-10-23

## Contexto
O sistema deve poder responder "desconhecida". O threshold controla o equilibrio entre FAR e FRR.

## Decisao
Escolher o threshold no conjunto de validacao (por exemplo, no ponto do EER ou com FAR alvo), nunca no teste. Valor e metrica usados ficam registrados em configs/ e neste ADR apos os experimentos.
