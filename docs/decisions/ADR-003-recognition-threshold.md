# ADR-003 - Limiar de reconhecimento

- Status: aceita
- Data: 2026-10-23

## Contexto
O sistema deve poder responder "desconhecida". O threshold controla o equilibrio entre FAR e FRR.
No E01, o threshold do ponto de EER por pares (0.4746) deixou passar 5.48% dos desconhecidos no
conjunto aberto. Na identificacao, um desconhecido e comparado com todas as pessoas cadastradas e
fica com a mais parecida, o que eleva a similaridade dos impostores.

## Decisao
Escolher o threshold no conjunto de validacao, nunca no teste, usando o conjunto aberto (pessoas
cadastradas e desconhecidas) e um FAR alvo.

Valor adotado: FAR alvo de 1%, similaridade de cosseno, galeria por prototipo, threshold 0.7160
(E02; o E03 repete o mesmo valor porque a melhor epoca foi o baseline).

Alternativas avaliadas no E02, todas escolhidas so na validacao:
- EER por pares: 0.4746
- FAR 5%: 0.5823
- FAR 2%: 0.6753

## Consequencias
- No teste (31 pessoas), com FAR 1%: 0% de desconhecidos aceitos (0 de 365 fotos), 2.46% de
  conhecidos rejeitados e 97.54% de identificacao correta. Zero em 365 fotos nao prova FAR zero.
- O sistema fica mais rigido: prefere responder "desconhecida" a aceitar a pessoa errada.
- O valor foi calibrado no LFW. Na Parte 2, com fotos de camera real, ele deve ser recalibrado
  pelo mesmo criterio (FAR alvo na validacao).
- Na Parte 2, o valor do threshold deve ficar em arquivo de configuracao em `configs/`, e nao
  fixo no codigo.

## Relacionado
Experimentos E01, E02 e E03 (ver README).