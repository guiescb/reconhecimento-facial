# ADR-002 - Armazenamento de embeddings

- Status: aceita (confirmada na implementacao da Parte 2)
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
- O SQLite nao aplica chaves estrangeiras por padrao: toda conexao executa
  `PRAGMA foreign_keys = ON`, senao a exclusao em cascata nao funciona.
- Toda conexao executa tambem `PRAGMA secure_delete = ON`: o conteudo apagado e sobrescrito com
  zeros no arquivo, em vez de apenas ficar marcado como livre.
- O nome da pessoa e unico sem diferenciar maiusculas, e o cadastro (pessoa, embeddings e fotos)
  e feito em uma unica transacao: se algo falha, nada fica no banco nem em disco.
- A exclusao de uma pessoa remove o cadastro, os embeddings e as fotos associadas, e o teste
  automatizado confere que nada sobra.

## Consequencias
Simples, sem servidor, facil de excluir uma pessoa. Busca por forca bruta e suficiente para poucos
cadastros. Ao fim da disciplina, o arquivo .db e as fotos devem ser apagados.
