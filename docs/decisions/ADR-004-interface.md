# ADR-004 - Interface da aplicacao

- Status: aceita
- Data: 2026-10-24

## Contexto
A Parte 2 exige uma interface (web, desktop, mobile ou aplicacao de camera) para cadastrar,
reconhecer e excluir pessoas, com entrada por camera. O foco do trabalho e o sistema de IA e a
engenharia de software, nao o desenvolvimento de telas. Os dados sao biometricos e nao podem sair
do computador.

## Decisao
Aplicacao web local em Streamlit (`app/streamlit_app.py`), separada da logica: a interface so
chama `FaceService`.

## Motivos
- `st.camera_input` e `st.file_uploader` dao entrada por camera e por arquivo com pouco codigo.
- Roda 100% local (nenhuma imagem sai do computador) e com estatisticas de uso desativadas
  (`.streamlit/config.toml`).
- A logica (`service.py`, `store.py`, `detection.py`) nao depende da interface nem do PyTorch,
  entao e testada com detector e embedder de mentira, sem fotos reais.
- Os modelos sao carregados uma unica vez (`st.cache_resource`).

## Consequencias
- A camera funciona por fotos, nao por video continuo.
- A interface nao tem testes automatizados; foi verificada manualmente.
- Desktop e mobile nao foram avaliados.
- Navegadores com traducao automatica ligada podem quebrar a tela do Streamlit (erro
  `removeChild`); basta desativar a traducao para localhost.
