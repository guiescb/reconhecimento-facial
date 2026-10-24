# ADR-002 - Armazenamento de embeddings

- Status: proposta (confirmar ao implementar a Parte 2)
- Data: 2026-10-23

## Contexto
Embeddings sao dado biometrico privado e precisam ser removiveis por pessoa.

## Decisao
SQLite local (arquivo .db, ignorado pelo Git), com tabela de pessoas e tabela de embeddings
ligadas por chave estrangeira com exclusao em cascata. Sem fotos armazenadas por padrao
(minimizacao); se armazenadas, ficam em diretorio privado e sao removidas junto com a pessoa.

Detalhes de implementacao:
- Cada pessoa tem varios embeddings (um por foto de cadastro). O prototipo usado no
  reconhecimento e calculado a partir deles, como no E02.
- O embedding e guardado como vetor de 512 valores normalizados.
- O banco fica em `data/private/`, pasta ignorada pelo Git (`*.db` tambem esta no `.gitignore`).
- O SQLite nao aplica chaves estrangeiras por padrao: a conexao deve executar
  `PRAGMA foreign_keys = ON`, senao a exclusao em cascata nao funciona.
- A exclusao de uma pessoa remove o cadastro, os embeddings e as fotos associadas, e o teste
  automatizado deve conferir que nada sobra.

## Consequencias
Simples, sem servidor, facil de excluir uma pessoa. Busca por forca bruta e suficiente para poucos
cadastros. Ao fim da disciplina, o arquivo .db e as fotos devem ser apagados.