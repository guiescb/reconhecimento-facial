# Guia do E02 (para estudar e defender)

## Pergunta do experimento
O E01 mostrou que o FaceNet separa muito bem pares de fotos (AUC 0.9995), mas na simulacao da
aplicacao (conjunto aberto) 5.48% dos desconhecidos foram aceitos como alguem cadastrado.
O E02 investiga **como cadastrar** e **onde cortar** para reduzir esse erro.

## O que muda em relacao ao E01
Os embeddings sao os mesmos (o E02 le o cache do E01, sem rodar o FaceNet de novo), assim como a
divisao treino/validacao/teste e o cadastro de 3 fotos por pessoa. So mudam duas coisas:

1. **Tipo de galeria**
   - *Prototipo*: cada pessoa vira a media (normalizada) das 3 fotos de cadastro. E o que o E01 ja fazia.
   - *Vizinho mais proximo (1-NN)*: guarda as 3 fotos separadas e usa a MAIS parecida com o rosto novo.
     Um desconhecido so precisa lembrar de UMA foto para ser aceito, entao tende a aceitar mais desconhecidos
     no mesmo threshold.
2. **Como escolher o threshold**
   - *EER por pares* (E01): threshold que equilibra FAR e FRR em pares de fotos da validacao. Nao olha o
     problema real (aceitar desconhecidos).
   - *FAR alvo no conjunto aberto*: na validacao, simula a aplicacao varias vezes (metade das pessoas
     cadastrada, metade desconhecida), junta a melhor similaridade dos desconhecidos e escolhe o menor
     threshold que deixa no maximo 5%, 2% ou 1% deles aceitos. Os alvos foram definidos antes de ver o teste.

## Regras de avaliacao respeitadas (topico 29)
- Threshold escolhido **so na validacao**. O teste apenas reporta.
- O grafico varre thresholds no teste so para mostrar a curva; nenhuma escolha depende dele.
- Identidades disjuntas entre treino, validacao e teste (checagem de leakage roda de novo).

## Como ler o resultado
- `id_correta`: conhecidos reconhecidos como a pessoa certa.
- `desc_aceito`: desconhecidos aceitos como alguem cadastrado (o erro mais grave).
- `FRR_conh`: conhecidos rejeitados por engano.
- `erro_pessoa`: conhecidos reconhecidos como OUTRA pessoa.
- `acc_aberto`: acertos totais (conhecidos certos + desconhecidos rejeitados).
- Endurecer o threshold troca `desc_aceito` menor por `FRR_conh` maior. Nao existe threshold sem custo;
  o ponto de operacao e uma decisao do sistema.

## Cuidados na interpretacao
- Poucas pessoas no teste (31, sendo ~15 desconhecidas): as fotos de uma mesma pessoa nao sao independentes.
  Por isso o JSON traz `unknown_identities_with_acceptance` (quantas pessoas desconhecidas tiveram ao menos
  uma foto aceita). Diferencas de 1 a 2 pontos percentuais nao sao conclusivas.
- O threshold calibrado na validacao pode nao dar exatamente o FAR alvo no teste (outras pessoas). Compare
  o alvo com o valor obtido e discuta a diferenca.
- Este experimento continua sem fine-tuning: nao muda o modelo, so a regra de decisao.
