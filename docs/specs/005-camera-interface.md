# 005 - Interface

Status: implementada

## Requisitos
- RF1: exibir a imagem da camera (ou arquivo) com caixa e nome da pessoa reconhecida ou "desconhecida" (verde = reconhecida, vermelho = desconhecida).
- RF2: acoes de cadastrar, reconhecer e excluir pessoa (paginas Reconhecer, Cadastrar, Pessoas e exclusao, Sobre).
- RF3: camera indisponivel: a interface oferece a opcao Arquivo e orienta a permitir o acesso a camera no navegador.
- RF4: erros tratados viram mensagens claras; erros inesperados vao para `logs/app.log` e a tela mostra um aviso generico, sem travar.

## Criterios de aceitacao
- Iniciar sem cadastros mostra "Nenhuma pessoa cadastrada".
- Camera indisponivel -> o usuario ainda consegue usar a aplicacao pela opcao Arquivo, com a orientacao exibida na tela.
- Falha ao carregar modelos ou configuracao -> mensagem de erro em vez de quebrar.
- A tecnologia de interface escolhida esta justificada no ADR-004.

## Implementacao
`app/streamlit_app.py` (Streamlit). A interface so chama `FaceService`; nao ha testes automatizados da tela, que foi verificada manualmente (cadastro, reconhecimento, desconhecido, erros e exclusao).
