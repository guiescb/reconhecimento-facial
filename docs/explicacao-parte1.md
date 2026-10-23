# Guia do codigo da Parte 1 (para estudar e defender)

## Ideia geral
O modelo (FaceNet) transforma cada rosto em um vetor de 512 numeros (embedding). Rostos da
mesma pessoa devem ficar proximos; de pessoas diferentes, distantes. Reconhecer = comparar vetores.

## Arquivos
- `data.py`: baixa o LFW direto da fonte (scikit-learn) e recorta/redimensiona para 160x160.
  Tambem gera embeddings sinteticos para testar sem dados reais.
- `embeddings.py`: passa as imagens pelo FaceNet em lotes (CPU) e normaliza os vetores (tamanho 1).
  Salva em cache para nao recalcular.
- `splits.py`: divide **pessoas** (nao fotos) em treino/validacao/teste. Como cada pessoa fica em um
  unico conjunto, nao ha foto igual nem parecida cruzando conjuntos (data leakage, topico 30).
- `metrics.py`: monta pares (mesma pessoa x pessoas diferentes), calcula similaridade
  (cosseno ou -distancia euclidiana), ROC, AUC, EER e as metricas no threshold.
- `identification.py`: simula a aplicacao real: algumas pessoas sao "cadastradas" (3 fotos -> media
  dos embeddings) e outras sao desconhecidas. Se a melhor similaridade < threshold -> desconhecida.
- `experiments/run_e01.py`: junta tudo e salva JSON + graficos em `results/`.

## Conceitos
- **FAR** (False Acceptance Rate): impostores aceitos / total de impostores.
- **FRR** (False Rejection Rate): pessoas legitimas rejeitadas / total de legitimas.
- **EER**: ponto onde FAR = FRR; quanto menor, melhor o modelo.
- **Threshold**: limiar de decisao. Regra: similaridade >= threshold -> reconhecida.
  Escolhido na **validacao** (no ponto do EER); o teste so reporta, nunca escolhe nada.
- **Cosseno x Euclidiana**: com vetores normalizados as duas ordenam os pares de forma muito
  parecida; por isso comparamos as duas como parte do experimento.
- **Sem fine-tuning**: os pesos do FaceNet (vggface2) nao sao alterados; e o baseline contra o qual
  os proximos experimentos (fine-tuning parcial/completo, outras perdas) serao comparados.
