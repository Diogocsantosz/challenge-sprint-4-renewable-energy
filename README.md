# Challenge Sprint 4 — Energia renovável

Neste projeto usamos Regressão Logística para prever a classe de energia (`Energy_Class`) do dataset *Renewable Energy Production Dataset (2010–2020)*. Também analisamos as correlações e comparamos o modelo com duas divisões de treino e teste.

## Integrantes

- Rafael Laprega Gontijo Magalhaes — RM 561975
- Gustavo Torres de Oliveira — RM 572952
- Lucas Furquim Lima — RM 568690
- Diogo Chiaradia Santos — RM 570246

## Dados e arquivos

O CSV usado está em `data/renewable_energy_production_dataset.csv`. Ele foi obtido no [Kaggle](https://www.kaggle.com/datasets/itsrohithere/renewable-energy-production-dataset-2010-2020). A página do Kaggle mostra a licença como “Unknown”; confira os termos de uso antes de redistribuir o arquivo publicamente. A atividade pede que a base esteja no repositório.

```text
data/       dataset utilizado
notebooks/  notebook da análise
src/        código Python
outputs/    gráficos, matrizes e métricas gerados pelo código
```

## O que fizemos

Codificamos `Energy_Class` com `LabelEncoder`. Para montar a matriz de correlação, mantivemos as colunas numéricas e transformamos as categóricas em indicadores one-hot. A matriz inclui o alvo codificado.

Como as classes são categorias nominais, os números atribuídos pelo encoder não representam uma ordem ou distância entre elas. Por isso, usamos a correlação com `Energy_Class` apenas como uma referência inicial, não como prova de que uma variável deve ser removida.

Nenhum par de atributos teve correlação absoluta igual ou superior a 0,85. Também não encontramos uma relação linear forte entre um preditor e o alvo: a maior correlação absoluta com o alvo foi cerca de 0,167, para `Efficiency_Ratio`. Com esses resultados, decidimos manter todos os preditores no modelo. Correlação, sozinha, não mostra se um atributo pode ajudar por meio de relações não lineares ou combinações com outros atributos.

Escolhemos Regressão Logística, um modelo linear. Para cada classe, ele calcula um escore da forma `wᵀx + b` e o transforma em uma probabilidade. A fronteira entre duas classes ocorre quando os escores são iguais; essa condição forma uma reta, um plano ou um hiperplano, dependendo do número de atributos. O código também usa imputação, padronização e one-hot encoding dentro do pipeline, ajustando essas etapas apenas com os dados de treino.

## Resultados

Usamos `random_state=42` nos dois cenários e estratificamos as divisões quando possível.

| Cenário | Treino | Teste | Acurácia | Baseline majoritária | Precisão weighted | Recall weighted | Precisão macro | Recall macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60% (600) | 40% (400) | 0,7200 | 0,4875 | 0,7083 | 0,7200 | 0,6938 | 0,6439 |
| 2 | 85% (850) | 15% (150) | 0,7933 | 0,4867 | 0,7927 | 0,7933 | 0,8000 | 0,7527 |

O dataset tem 1.000 registros: 486 da classe High, 336 de Medium e 178 de Low. Nos dois cenários, a acurácia foi maior que a baseline majoritária, que simplesmente prevê a classe mais frequente.

O resultado não é igual para todas as classes. No cenário 60/40, o recall foi 0,9436 para High, 0,4507 para Low e 0,5373 para Medium. No cenário 85/15, foi 0,9315 para High, 0,6667 para Low e 0,6600 para Medium. Assim, o modelo é um ponto de partida razoável, mas ainda erra mais as classes Low e Medium. A adequação depende também do impacto que esses erros teriam no uso real.

O teste com 40% dos dados tem mais exemplos para avaliar o modelo, mas deixa menos dados para treinamento. Com 15% de teste, o modelo treina com mais registros, mas as métricas dependem de uma amostra menor. Neste experimento, o segundo cenário teve resultados maiores; isso não prova que seja sempre melhor, pois os tamanhos dos testes são diferentes. Validação cruzada poderia ajudar a comparar os cenários com mais estabilidade.

## Como executar

Com Python instalado, abra um terminal na pasta do projeto e rode:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/analysis.py --no-show
```

O CSV precisa estar no caminho indicado acima. Para usar outro caminho, informe-o com `--data`:

```bash
python src/analysis.py --data /caminho/para/arquivo.csv --no-show
```

Também é possível abrir `notebooks/challenge_sprint_4.ipynb` e executar as células a partir da raiz do projeto.

## Arquivos gerados

- `outputs/correlation_heatmap.png` e `outputs/correlation_matrix.csv`: correlações;
- `outputs/confusion_matrix_cenário_1.png` e `outputs/confusion_matrix_cenário_2.png`: matrizes de confusão;
- `outputs/metrics_comparison.csv`: métricas dos dois cenários;
- `outputs/classification_reports.txt`: métricas detalhadas por classe.

O código usa a semente 42 para permitir repetir a divisão dos dados. A Regressão Logística pode não representar bem fronteiras de decisão mais complexas; além disso, as métricas variam conforme os dados de teste. Por isso, os resultados devem ser interpretados junto com as matrizes de confusão e as métricas por classe, não apenas pela acurácia.
