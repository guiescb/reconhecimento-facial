# ADR-002 - Armazenamento de embeddings

- Status: proposta
- Data: 2026-10-23

## Contexto
Embeddings sao dado biometrico privado e precisam ser removiveis por pessoa.

## Decisao
SQLite local (arquivo .db, ignorado pelo Git), com tabela de pessoas e tabela de embeddings ligadas por chave estrangeira com exclusao em cascata. Sem fotos armazenadas por padrao (minimizacao); se armazenadas, ficam em diretorio privado e sao removidas junto com a pessoa.

## Consequencias
Simples, sem servidor, facil de excluir uma pessoa. Busca por forca bruta e suficiente para poucos cadastros.
