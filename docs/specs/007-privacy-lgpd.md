# 007 - Privacidade e LGPD

Status: rascunho (atualizar conforme a implementacao)

## Requisitos
- RF1: nenhum dado privado (fotos, embeddings, bancos, checkpoints derivados, termos) no Git; .gitignore criado antes do primeiro commit.
- RF2: dados privados processados apenas localmente; agentes de IA e servicos externos recebem so codigo, specs e dados sinteticos.
- RF3: participantes adultos, com Termo de Consentimento (Anexo A) guardado fora do repositorio.
- RF4: resultados com dados privados somente agregados e anonimos.

## Criterios de aceitacao
- python scripts/check_repo_privacy.py retorna OK.
- Checklist do Anexo B totalmente marcado antes da entrega.
