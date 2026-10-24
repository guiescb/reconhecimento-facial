# Guia do E03 (para estudar e defender)

## Pergunta
Treinar uma camada por cima do FaceNet melhora o reconhecimento em relacao ao baseline
(E01/E02)? Esta e a parte de **treinamento/fine-tuning parcial** do enunciado (topicos 27 e 28).

## O que e treinado e o que nao e
- O FaceNet **nao muda** (pesos congelados). Os embeddings de 512 numeros ja estao em cache.
- Treinamos uma camada linear 512 -> 512 (cerca de 262 mil parametros) que transforma cada
  embedding e normaliza o resultado. Ela **comeca como a matriz identidade**, entao a epoca 0 e
  exatamente o baseline. Isso garante um ponto de comparacao justo.
- So as **95 identidades de treino** entram no treino. Validacao (32) e teste (31) sao outras pessoas.

## As duas funcoes de perda
- **Triplet Loss (batch-hard):** cada lote tem P pessoas x K fotos. Para cada foto (ancora) pegamos
  o positivo mais distante e o negativo mais proximo, e exigimos que o negativo fique mais longe
  que o positivo por uma margem (0.3). Ensina a *distancia* diretamente (metric learning).
- **Cross-Entropy com classificador por cosseno:** a camada tenta classificar as 95 pessoas de
  treino (logits = 16 x cosseno com o vetor de cada classe). Ensina a *separar classes* e o
  embedding aparece como subproduto.

## Como a melhor epoca e escolhida
A cada epoca medimos o **EER de pares na validacao**. Ficamos com a epoca de menor EER (a epoca 0,
o baseline, tambem concorre). O teste so e usado no fim, para reportar. Assim o treino nao
"olha" o teste (topico 29).

## Por que o resultado pode ser pequeno ou ate negativo
O baseline ja chega a AUC 0.9995, entao ha pouco a ganhar. Alem disso, so ha 95 pessoas e a camada
tem muitos parametros: ela pode **decorar o treino** (perda de treino perto de zero enquanto o
EER da validacao piora). Isso e overfitting, e o criterio de validacao existe para pegar isso.
Reportar um ganho nulo ou negativo com essa explicacao e uma analise valida (topico 33).

## Como comparar
A tabela do terminal usa a mesma avaliacao para o baseline e para cada camada: pares (AUC, EER,
F1, FAR, FRR com threshold da validacao) e conjunto aberto (identificacao correta e
desconhecidos aceitos, com thresholds calibrados na validacao). A linha "baseline" deve
repetir os numeros do prototipo no E02. O ponto principal e FAR 1%, definido antes de ver o teste.

## Nota sobre os dados
Os pesos da camada treinada ficam em `data/cache/` (ignorado pelo Git). Foram treinados so com o
dataset publico (LFW), mas continuam fora do repositorio; quem reproduzir roda o `run_e03.py`.
