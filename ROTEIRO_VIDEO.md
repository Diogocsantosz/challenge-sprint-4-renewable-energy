# Roteiro de apresentação — vídeo não listado

**Duração sugerida:** 5–7 minutos. Usem este roteiro como guia, não como texto para decorar. Cada integrante pode explicar a parte que entende melhor. As métricas abaixo vêm da execução do projeto.

## 1. Abertura (20–30 s)

“Oi! Somos Rafael Laprega Gontijo Magalhaes (RM 561975), Gustavo Torres de Oliveira (RM 572952), Lucas Furquim Lima (RM 568690) e Diogo Chiaradia Santos (RM 570246). Neste trabalho usamos dados de energia renovável para prever a classe `Energy_Class`. Também comparamos a Regressão Logística em dois cenários de treino e teste.”

## 2. Dados e preparação (45–60 s)

Mostrar o CSV em `data/` e a leitura dos dados. “A base tem 1.000 linhas. O código verifica os tipos das colunas, valores ausentes e quantas linhas existem em cada classe. Transformamos `Energy_Class` em números para treinar o modelo. As outras colunas categóricas são convertidas em indicadores pelo pipeline.”

## 3. Correlação e decisão de features (60 s)

Mostrar o heatmap. “Não apareceu nenhum par de atributos com correlação absoluta de 0,85 ou mais. A maior correlação com o alvo foi cerca de 0,167. Como os números usados para codificar as classes não significam que uma classe seja maior ou menor que outra, usamos essa correlação apenas como referência. Por isso, mantivemos todos os preditores.”

## 4. Modelo linear (45–60 s)

Mostrar a função `build_model`. “Escolhemos Regressão Logística. Apesar do nome, usamos ela para classificação. O modelo calcula um escore linear para cada classe e escolhe a classe com maior probabilidade. Por isso, a fronteira entre classes é linear no espaço dos atributos. O pipeline também prepara os dados usando apenas o conjunto de treino.”

## 5. Cenários e resultados (90–120 s)

Mostrar as duas matrizes de confusão e `metrics_comparison.csv`. “No primeiro cenário, usamos 60% dos dados para treino e 40% para teste. No segundo, usamos 85% para treino e 15% para teste. Mantivemos a mesma semente para tornar a divisão reproduzível.”

“No cenário 60/40, a acurácia foi **0,7200**, a precisão ponderada **0,7083** e o recall ponderado **0,7200**. No cenário 85/15, foram **0,7933**, **0,7927** e **0,7933**. A baseline, que sempre prevê a classe mais frequente, ficou perto de **0,487** nos dois casos. No cenário 60/40, o recall de Low foi **0,4507**; no 85/15, os recalls de Low e Medium foram **0,6667** e **0,6600**. As matrizes ajudam a ver onde aconteceram as confusões.”

“O modelo foi melhor que a baseline, mas ainda confunde algumas classes, principalmente Low e Medium. Com 40% de teste temos mais exemplos para avaliar, mas menos dados para treinar. Com 15% de teste treinamos com mais dados, mas a avaliação fica mais dependente de uma amostra menor. Então não dá para dizer que um cenário é definitivamente melhor só olhando esses números.”

## 6. Fechamento (20–30 s)

“Os gráficos e os resultados que mostramos ficam na pasta `outputs/`. Como próximo passo, poderíamos testar o modelo com validação cruzada para ter uma comparação mais estável. Obrigado!”

## Checklist antes de gravar

- [ ] Mostrar o CSV em `data/` e confirmar que o script executou sem erros.
- [ ] Conferir as métricas citadas com `outputs/metrics_comparison.csv`.
- [ ] Mostrar heatmap, as duas matrizes de confusão e a tabela comparativa.
- [ ] Não divulgar dados pessoais nem credenciais. Confira os termos do dataset (a licença exibida no Kaggle é “Unknown”) antes de publicar a base.
- [ ] Conferir áudio, legibilidade do terminal e compartilhamento de tela.
