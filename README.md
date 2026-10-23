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

## Desenvolvimento
Spec-Driven Development: requisitos e criterios de aceitacao em `docs/specs/`,
decisoes em `docs/decisions/`. Agentes de IA recebem apenas codigo, specs e dados sinteticos
(nunca fotos ou embeddings reais).

## Checagem de privacidade antes de entregar
```
python scripts/check_repo_privacy.py
```
