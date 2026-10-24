# 006 - Exclusao de pessoa (Privacy by Design)

Status: implementada

## Requisitos
- RF1: excluir uma pessoa remove cadastro, embedding, fotografias armazenadas e demais dados ligados a identidade.
- RF2: logs nao guardam nome nem dados biometricos (so ids curtos e contagens).
- RF3: a exclusao e completa no arquivo do banco: `PRAGMA foreign_keys = ON` (cascata) e `PRAGMA secure_delete = ON` em toda conexao.
- RF4: a interface pede confirmacao antes de excluir.

## Criterios de aceitacao
- Apos excluir, a pessoa nao e mais reconhecida e nao ha arquivos ou linhas associadas a ela.
- Excluir pessoa inexistente retorna erro tratado (PersonNotFound).
- Teste automatizado verifica a remocao completa e que os logs nao contem nomes.

## Implementacao
`src/facerec/store.py` (`delete_person`), `src/facerec/service.py`; testes em `tests/test_store.py` e `tests/test_service.py`.
