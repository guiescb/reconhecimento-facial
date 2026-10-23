# facerec - Reconhecimento Facial (IA - UFMS)

Trabalho da disciplina de Inteligencia Artificial. Parte 1: experimentacao e fine-tuning.
Parte 2: aplicacao de cadastro e reconhecimento facial.

> Privacidade: este repositorio NAO contem fotos, embeddings, bancos, checkpoints treinados
> com dados privados nem termos de consentimento. Veja `data/README.md` e `.gitignore`.

## Estrutura
```
src/facerec/    codigo-fonte         experiments/    scripts dos experimentos
configs/        configuracoes        tests/          testes automatizados
docs/specs/     especificacoes       docs/decisions/ ADRs
data/README.md  origem dos dados     scripts/        utilitarios (checagem de privacidade)
```

## Ambiente
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```
Hardware usado nos experimentos: notebook, CPU, Intel HD Graphics 520 (sem GPU dedicada).

## Experimentos (Parte 1)
| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|---|---|---|---|---|---|---|
| E01 | FaceNet (VGGFace2) | baseline, cosseno | LFW | 0.9925 | 0.9925 | Accuracy de pares. AUC 0.9995, EER 0.0068; conjunto aberto: 5.48% de desconhecidos aceitos |
| E01 | FaceNet (VGGFace2) | baseline, euclidiana | LFW | 0.9925 | 0.9925 | Idêntico ao cosseno (embeddings normalizados) |
| E02 | FaceNet (VGGFace2) | protótipo, threshold FAR 1% calibrado na validação | LFW | 0.9827 | — | Accuracy do conjunto aberto. Ponto principal (definido a priori). Id. correta 97.54%, desconhecidos aceitos 0% |
| E02 | FaceNet (VGGFace2) | vizinho mais próximo, threshold FAR 1% | LFW | 0.9655 | — | Accuracy do conjunto aberto. Id. correta 95.07%, desconhecidos aceitos 0% |

Resultados com dados privados aparecem somente agregados, sem identificar participantes.

## Análise do E01: baseline sem fine-tuning

**Hipótese.** O FaceNet (InceptionResnetV1, pesos VGGFace2) pré-treinado, sem nenhum treino adicional, já gera embeddings que separam bem as pessoas do LFW. Esperávamos EER baixo e resultado parecido entre cosseno e distância euclidiana.

**Configuração.** LFW com identidades de pelo menos 10 fotos: 4324 imagens e 158 identidades. A divisão é por identidade (60/20/20): treino 95 pessoas (2115 imagens), validação 32 (944) e teste 31 (1265). Nenhuma pessoa aparece em dois conjuntos, e a checagem de leakage passou. Embeddings de 512 dimensões, normalizados. Foram usados 4000 pares na validação e 4000 no teste, metade da mesma pessoa. Seed 42, Python 3.14.7, torch 2.14.1 (CPU), notebook com 8 GB de RAM, sem GPU. Extrair os embeddings levou 657 s, uma vez só.

**Resultados.** O threshold foi escolhido na validação (EER 0,60%, threshold 0,4746). No teste:

| Medida | Valor |
|---|---|
| AUC | 0.9995 |
| EER no teste | 0.68% |
| Accuracy / F1 | 0.9925 / 0.9925 |
| FAR / FRR | 0.35% (7 de 2000) / 1.15% (23 de 2000) |

![ROC do E01](results/E01_roc.png)

![Distribuição de similaridade do E01](results/E01_distribuicoes.png)

No conjunto aberto (852 fotos de pessoas cadastradas e 365 de desconhecidas), 99.88% dos conhecidos foram identificados corretamente, nenhum foi confundido com outra pessoa, e 5.48% dos desconhecidos (20 de 365) foram aceitos como alguém cadastrado.

**Discussão.**
- **Cosseno e euclidiana deram resultados idênticos.** Os embeddings têm norma 1, então as duas medidas ordenam os pares do mesmo jeito (a distância ao quadrado é 2 − 2·cosseno). Comparar as duas não traz informação nova, e essa comparação só se torna informativa com outros classificadores.
- **O threshold da validação generalizou bem.** O ponto de EER no teste ficaria em 0,4200, mas com o threshold da validação o erro ficou próximo. O sistema ficou um pouco mais rígido que o ideal (FRR maior que FAR).
- **O ponto fraco é a identificação de desconhecidos.** Os 5.48% de desconhecidos aceitos motivaram o E02.

**Limitações.** O FaceNet foi pré-treinado no VGGFace2 e pode haver identidades em comum com o LFW, então o resultado pode estar otimista. O LFW é um benchmark relativamente fácil, e fotos reais de câmera devem render menos. O subconjunto de 158 identidades foi escolhido para caber em CPU e 8 GB de RAM.

## Análise do E02: conjunto aberto

Resultados no conjunto de teste (31 pessoas, metade cadastrada e metade desconhecida). O threshold de cada linha foi escolhido só na validação.

| Galeria | Threshold | Valor | Id. correta | Desconhecidos aceitos | Conhecidos rejeitados | Acurácia do conjunto aberto |
|---|---|---|---|---|---|---|
| Protótipo | EER por pares (E01) | 0.4746 | 99.88% | 5.48% | 0.12% | 98.27% |
| Protótipo | FAR 5% | 0.5823 | 99.77% | 0.27% | 0.23% | 99.75% |
| Protótipo | FAR 2% | 0.6753 | 98.71% | 0.00% | 1.29% | 99.10% |
| Protótipo | FAR 1% (principal) | 0.7160 | 97.54% | 0.00% | 2.46% | 98.27% |
| Vizinho mais próximo | EER por pares (E01) | 0.4746 | 99.88% | 10.14% | 0.12% | 96.88% |
| Vizinho mais próximo | FAR 5% | 0.5978 | 99.88% | 0.55% | 0.12% | 99.75% |
| Vizinho mais próximo | FAR 2% | 0.6962 | 97.77% | 0.00% | 2.23% | 98.44% |
| Vizinho mais próximo | FAR 1% (principal) | 0.7406 | 95.07% | 0.00% | 4.93% | 96.55% |

![Conjunto aberto no teste](results/E02_conjunto_aberto.png)

O gráfico da esquerda mostra a identificação correta contra os desconhecidos aceitos. Os círculos são os thresholds escolhidos na validação e o círculo vazado é o do E01. O da direita mostra a melhor similaridade de conhecidos e desconhecidos, com o threshold do E01 e o de FAR 1%.

**Discussão.** Em verificação de pares o baseline já era muito bom (AUC 0.9995). O ponto fraco aparece na identificação, em que um desconhecido é comparado com todas as pessoas cadastradas e fica com a mais parecida. Isso eleva a similaridade dos impostores, e o threshold de EER por pares deixava passar 5.48% dos desconhecidos. Calibrar o threshold no conjunto aberto da validação reduziu esse erro para 0.27% (FAR 5%) com perda de 0.11 ponto na identificação correta. O vizinho mais próximo aceitou mais desconhecidos no threshold do E01 e identificou pior nos thresholds mais rígidos, mas as diferenças são pequenas e o teste tem poucas pessoas desconhecidas, então não são conclusivas.

**Limitações.** O threshold foi calibrado no LFW e deve ser recalibrado com fotos de câmera real. Os alvos de FAR da validação ficaram conservadores no teste (0% observado), e 0 em 365 fotos não prova FAR zero.

## Desenvolvimento
Spec-Driven Development: requisitos e criterios de aceitacao em `docs/specs/`,
decisoes em `docs/decisions/`. Agentes de IA recebem apenas codigo, specs e dados sinteticos
(nunca fotos ou embeddings reais).

## Checagem de privacidade antes de entregar
```
python scripts/check_repo_privacy.py
```
