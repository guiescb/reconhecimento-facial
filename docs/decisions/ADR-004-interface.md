# ADR-004 - Interface da aplicação

**Status:** aceita

## Decisão
Aplicação web local em Streamlit (`app/streamlit_app.py`), separada da lógica: a interface só chama `FaceService`.

## Justificativa
- Entrada por câmera e por arquivo com pouco código, o que permite focar nos requisitos.
- Roda 100% local (nenhuma imagem sai do computador) e com estatísticas de uso desativadas.
- A lógica (`service.py`, `store.py`, `detection.py`) não depende da interface nem do PyTorch, então é testada com dublês sem fotos reais.
