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
| Experimento | Modelo | Estrategia | Dataset | Accuracy | F1 | Observacao |
|---|---|---|---|---|---|---|
| E01 | a definir | baseline (sem fine-tuning) | publico | ... | ... | ... |
| E02 | a definir | fine-tuning parcial | publico | ... | ... | ... |
| E03 | a definir | fine-tuning parcial | publico | ... | ... | ... |

Resultados com dados privados aparecem somente agregados, sem identificar participantes.

## Desenvolvimento
Spec-Driven Development: requisitos e criterios de aceitacao em `docs/specs/`,
decisoes em `docs/decisions/`. Agentes de IA recebem apenas codigo, specs e dados sinteticos
(nunca fotos ou embeddings reais).

## Checagem de privacidade antes de entregar
```
python scripts/check_repo_privacy.py
```
