# 006 - Exclusao de pessoa (Privacy by Design)

Status: rascunho (atualizar conforme a implementacao)

## Requisitos
- RF1: excluir uma pessoa remove cadastro, embedding, fotografias armazenadas e demais dados ligados a identidade.
- RF2: logs nao guardam nome nem dados biometricos.

## Criterios de aceitacao
- Apos excluir, a pessoa nao e mais reconhecida e nao ha arquivos ou linhas associadas a ela.
- Excluir pessoa inexistente retorna erro tratado.
- Teste automatizado verifica a remocao completa.
