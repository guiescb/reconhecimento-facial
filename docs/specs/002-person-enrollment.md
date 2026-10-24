# 002 - Cadastro de pessoa

Status: implementada

## Requisitos
- RF1: Pessoa -> Imagem -> Deteccao -> Alinhamento -> Embedding -> Cadastro.
- RF2: rejeitar imagem sem face, com multiplas faces ou inadequada (muito pequena ou borrada).
- RF3: impedir cadastro duplicado (embedding muito similar a pessoa ja cadastrada) e nome repetido (sem diferenciar maiusculas).
- RF4: armazenar apenas o necessario (minimizacao de dados), localmente: id, nome e embeddings; fotos so se `store_photos: true` (desligado por padrao).
- RF5: exigir de 3 a 10 fotos por pessoa (o mesmo numero de fotos de cadastro dos experimentos E01 a E03) e nome nao vazio.
- RF6: a interface exige a confirmacao de que a pessoa e maior de 18 anos e assinou o Termo de Consentimento (Anexo A, guardado fora do repositorio).
- RF7: o cadastro e atomico: qualquer falha nao deixa nada no banco nem em disco.

## Criterios de aceitacao
- Cadastro valido cria um registro com identificador e um embedding por foto.
- Cada caso de rejeicao retorna mensagem clara, indica o numero da foto com problema ("Foto 3: ...") e nao altera o banco.
- Quantidade de fotos fora de 3 a 10 -> InvalidPhotoCount; nome vazio -> InvalidName.
- Segundo cadastro da mesma pessoa e detectado como duplicado (DuplicateEnrollment); nome repetido -> PersonAlreadyExists.
- Cadastro sem marcar o consentimento e bloqueado na interface.
- Banco inicial vazio: a aplicacao informa "Nenhuma pessoa cadastrada".

## Implementacao
`src/facerec/service.py` (`enroll`), `src/facerec/store.py` (`add_person`), `app/streamlit_app.py`; testes em `tests/test_service.py` e `tests/test_store.py`.
