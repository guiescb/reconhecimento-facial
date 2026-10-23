# 002 - Cadastro de pessoa

Status: rascunho (atualizar conforme a implementacao)

## Requisitos
- RF1: Pessoa -> Imagem -> Deteccao -> Alinhamento -> Embedding -> Cadastro.
- RF2: rejeitar imagem sem face, com multiplas faces ou inadequada (muito pequena ou borrada).
- RF3: impedir cadastro duplicado (embedding muito similar a pessoa ja cadastrada).
- RF4: armazenar apenas o necessario (minimizacao de dados), localmente.

## Criterios de aceitacao
- Cadastro valido cria um registro com identificador e embedding.
- Cada caso de rejeicao retorna mensagem clara e nao altera o banco.
- Segundo cadastro da mesma pessoa e detectado como duplicado.
- Banco inicial vazio: a aplicacao informa "Nenhuma pessoa cadastrada".
