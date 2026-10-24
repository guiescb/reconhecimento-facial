# 007 - Privacidade e LGPD

Status: implementada (o checklist do Anexo B e marcado na entrega)

## Requisitos
- RF1: nenhum dado privado (fotos, embeddings, bancos, checkpoints derivados, termos) no Git; .gitignore criado antes do primeiro commit.
- RF2: dados privados processados apenas localmente; agentes de IA e servicos externos recebem so codigo, specs e dados sinteticos (os testes usam `tests/fakes.py`).
- RF3: participantes adultos, com Termo de Consentimento (Anexo A) guardado fora do repositorio; a interface exige a confirmacao no cadastro.
- RF4: resultados com dados privados somente agregados e anonimos.
- RF5: telemetria do Streamlit desativada (`.streamlit/config.toml`).

## Criterios de aceitacao
- python scripts/check_repo_privacy.py retorna OK (arquivos atuais e historico).
- Checklist do Anexo B totalmente marcado antes da entrega.
- Quem clona o repositorio encontra "Nenhuma pessoa cadastrada" e cria o proprio banco local.
