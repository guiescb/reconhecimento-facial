# Dados

Este diretorio NAO contem dados. Tudo aqui (exceto este arquivo) e ignorado pelo Git.

## Dataset publico

| Item | Informacao |
|---|---|
| Nome | LFW - Labeled Faces in the Wild |
| Fonte oficial | http://vis-www.cs.umass.edu/lfw/ (consultada via copia da Wayback Machine, pois o site estava sobrecarregado) |
| Finalidade | Experimentacao e benchmark de reconhecimento facial |
| Licenca / termos | A pagina oficial do LFW (copia da Wayback Machine: https://web.archive.org/web/20241222121610/http://vis-www.cs.umass.edu/lfw/, datada de 22/12/2024) nao declara licenca nem termos de uso nas secoes Pagina inicial, Download e Informacao. Nenhuma licenca foi inferida. O dataset foi divulgado para estudo de reconhecimento facial; citar Huang et al. (2007), UMass TR 07-49. Uso neste trabalho: estritamente academico e local, sem redistribuicao. |
| Uso neste projeto | Treino/validacao/teste dos experimentos da Parte 1 |
| Redistribuicao | Nao incluido no repositorio |

Como obter: o codigo usa fetch_lfw_people do scikit-learn, que baixa uma copia do arquivo hospedada no figshare para a pasta de dados do scikit-learn, fora do repositorio. O dataset original esta na fonte oficial acima.

## Dataset privado

Fotos de participantes adultos que assinaram o Termo de Consentimento (Anexo A).
Fica apenas na maquina do autor, em `data/private/`, e NUNCA e versionado.
Estrutura local criada pela aplicação: `data/private/facerec.db` (cadastros e embeddings) e, só se `store_photos: true`, `data/private/photos/<id>/01.jpg`.
Fotografias, nomes, embeddings e o banco são todos dados biométricos privados.
Os termos assinados ficam em `consent/`, que o Git ignora.
