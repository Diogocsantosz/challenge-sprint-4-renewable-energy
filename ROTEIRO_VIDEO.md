# Roteiro de apresentação — vídeo não listado

**Duração sugerida:** 5–7 minutos. **Formato:** compartilhamento de tela com o código e resultados gerados. As métricas abaixo foram preenchidas a partir da execução do projeto.

## 1. Abertura (20–30 s)

“Olá! Neste vídeo apresento o Challenge Sprint 4. O objetivo é classificar `Energy_Class` usando o dataset Renewable Energy Production Dataset de 2010 a 2020 e comparar dois tamanhos de conjunto de teste com Regressão Logística.”

## 2. Dados e preparação (45–60 s)

Mostrar o CSV na pasta `data/` e a leitura/inspeção no notebook. “O código verifica dimensões, tipos, valores ausentes e distribuição das classes. A variável-alvo é codificada com `LabelEncoder`; o mapeamento fica impresso para rastreabilidade. As colunas categóricas dos preditores são tratadas com one-hot encoding no pipeline, não como números ordinais.”

## 3. Correlação e decisão de features (60 s)

Mostrar o heatmap e as correlações altas impressas. “A matriz inclui todos os preditores transformados e `Energy_Class` codificada. A codificação de uma classe nominal não tem ordem semântica, então correlação é apenas exploratória. Mantive todas as features como baseline e só consideraria remover variáveis após verificar redundância, vazamento, identificadores e conhecimento do domínio.”

## 4. Modelo linear (45–60 s)

Mostrar a função `build_model`. “A Regressão Logística calcula escores lineares, `wᵀx + b`, e os transforma em probabilidades. Em multiclasses, a fronteira entre duas classes satisfaz `(wᵢ-wⱼ)ᵀx + (bᵢ-bⱼ)=0`, um hiperplano no espaço das features. Imputação, padronização e codificação são ajustadas dentro do pipeline apenas no treino.”

## 5. Cenários e resultados (90–120 s)

Mostrar as duas matrizes de confusão e `metrics_comparison.csv`. “O primeiro cenário usa 60% para treino e 40% para teste; o segundo, 85% para treino e 15% para teste. Ambos mantêm `random_state=42` e usam estratificação quando viável.”

“No cenário 60/40, obtivemos acurácia de **0,7200**, precisão ponderada de **0,7083** e recall ponderado de **0,7200**. No cenário 85/15, os valores foram **0,7933**, **0,7927** e **0,7933**. No relatório por classe, o recall mais baixo no cenário 60/40 foi para Low (**0,4507**); no cenário 85/15, High teve recall **0,9315**, enquanto Low e Medium ficaram em **0,6667** e **0,6600**. As matrizes de confusão mostram em quais classes ocorrem esses erros.”

“Não concluo adequação apenas pela acurácia: observo desempenho por classe, distribuição, erros relevantes e baseline. O conjunto de teste maior estima métricas com mais observações, mas deixa menos dados para treino. O menor aumenta o treino, porém torna a avaliação mais sensível à amostra. As diferenças devem ser interpretadas com cautela.”

## 6. Fechamento (20–30 s)

“Os resultados são reproduzíveis e os gráficos, métricas e relatórios ficam na pasta `outputs/`. Para uma avaliação mais estável, o próximo passo seria validação cruzada estratificada e uma análise mais aprofundada dos erros. Obrigado!”

## Checklist antes de gravar

- [ ] Mostrar o CSV em `data/` e confirmar que o script executou sem erros.
- [ ] Conferir as métricas citadas com `outputs/metrics_comparison.csv`.
- [ ] Mostrar heatmap, as duas matrizes de confusão e a tabela comparativa.
- [ ] Não divulgar dados pessoais nem credenciais. Confira os termos do dataset (a licença exibida no Kaggle é “Unknown”) antes de publicar a base.
- [ ] Conferir áudio, legibilidade do terminal e compartilhamento de tela.
