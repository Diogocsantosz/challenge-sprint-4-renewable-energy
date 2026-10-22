# CHALLENGE SPRINT 4 — Renewable Energy Production

Classificação supervisionada da variável `Energy_Class` usando **Regressão Logística** e o dataset *Renewable Energy Production Dataset (2010–2020)*, do Kaggle. O projeto calcula correlações, treina e compara dois cenários de holdout e exporta gráficos e métricas.

> **Dados:** o CSV usado está em `data/renewable_energy_production_dataset.csv` e foi obtido do [Kaggle](https://www.kaggle.com/datasets/itsrohithere/renewable-energy-production-dataset-2010-2020). A página do dataset informa licença “Unknown”; confira os termos de uso antes de redistribuí-lo publicamente. A atividade exige que a base esteja no repositório.

## Estrutura

```text
challenge-sprint-4/
├── data/
│   └── renewable_energy_production_dataset.csv  # adicionar localmente
├── notebooks/
│   └── challenge_sprint_4.ipynb
├── outputs/                                      # gerado na execução
├── src/
│   └── analysis.py
├── requirements.txt
└── README.md
```

## Métodos e justificativas

### Preparação e correlação

O alvo `Energy_Class` é transformado em códigos inteiros com `LabelEncoder`; o mapeamento aparece no console. Para a análise exploratória, colunas categóricas dos preditores são expandidas em indicadores one-hot, e os valores numéricos são preservados. O heatmap contém esses atributos numéricos e o alvo codificado. Atenção: Pearson sobre códigos categóricos nominais pode sugerir associação que depende da codificação, e a numeração das classes não expressa distância real entre elas. Para alvos multiclasse, as correlações com o rótulo devem ser vistas como triagem exploratória, não como medida definitiva de importância.

O baseline utiliza todos os preditores, exceto `Energy_Class`; remove apenas colunas totalmente vazias. A decisão é apropriada sem evidência empírica suficiente para descartar variáveis. Pares com `|r| >= 0,85` são reportados para inspeção: redundância, vazamento de informação, identificadores e relevância de domínio devem orientar seleção futura. Correlação baixa não implica falta de poder preditivo, pois relações não lineares ou interações podem existir. Com Regressão Logística, regularização ajuda a limitar coeficientes, mas não torna variáveis irrelevantes automaticamente.

### Por que Regressão Logística é linear?

Para cada classe `k`, o modelo calcula `z_k = w_k^T x + b_k`; a função logística/softmax transforma esses escores em probabilidades. A fronteira de decisão entre as classes `i` e `j` é definida por `z_i = z_j`, isto é, `(w_i - w_j)^T x + (b_i - b_j) = 0`. Essa equação descreve uma reta, plano ou hiperplano (conforme a dimensão dos atributos). A fronteira é linear no espaço das features transformadas. O pipeline usa imputação, padronização numérica e one-hot encoding para categóricas, todos ajustados exclusivamente no treino de cada cenário.

### Divisão e métricas

Ambos os cenários usam `random_state=42` e tentam manter a proporção de classes com estratificação:

| Cenário | Treino | Teste |
|---|---:|---:|
| 1 | 60% | 40% |
| 2 | 85% | 15% |

Para problema multiclasse, precisão e recall `weighted` são exibidos como medidas globais, juntamente com as versões `macro` e o `classification_report` por classe. Isso evita ocultar classes minoritárias por trás da acurácia. A matriz de confusão usa os nomes originais das classes.

#### Resultados obtidos

| Cenário | N treino | N teste | Acurácia | Baseline majoritária | Precisão weighted | Recall weighted | Precisão macro | Recall macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60/40 | 600 | 400 | 0,7200 | 0,4875 | 0,7083 | 0,7200 | 0,6938 | 0,6439 |
| 85/15 | 850 | 150 | 0,7933 | 0,4867 | 0,7927 | 0,7933 | 0,8000 | 0,7527 |

O dataset contém 1.000 registros e as classes estão distribuídas em High (486), Medium (336) e Low (178). Nenhum par de atributos atingiu correlação absoluta de 0,85; portanto, não há evidência nessa análise de redundância linear forte que justifique remover features. A correlação de Pearson com `Energy_Class` codificada é fraca (maior valor absoluto: aproximadamente 0,167 para `Efficiency_Ratio`) e deve ser interpretada apenas como triagem, pois os números atribuídos às classes nominais não representam uma ordem real. Por isso, o modelo baseline utiliza todos os preditores.

Nos dois cenários, a acurácia supera a baseline majoritária (aproximadamente 0,487), indicando que o modelo aprendeu padrões além de sempre prever a classe mais frequente. O cenário 85/15 apresentou métricas maiores, mas sua avaliação usa apenas 150 registros e pode variar mais com a composição da amostra. Os recalls por classe mostram que o desempenho não é uniforme: no cenário 60/40, o recall foi 0,9436 para High, 0,4507 para Low e 0,5373 para Medium; no cenário 85/15, foi 0,9315, 0,6667 e 0,6600, respectivamente. Assim, o resultado é promissor como baseline, mas a adequação depende do custo dos erros, especialmente para as classes Low e Medium. A diferença entre os cenários não prova, por si só, que um seja superior; validação cruzada poderia fornecer uma comparação mais estável.

## Interpretação crítica esperada

- **Teste maior (40%)**: fornece mais observações para estimar as métricas e tende a reduzir a variância amostral da avaliação; em contrapartida, deixa menos dados para ajustar o modelo, o que pode reduzir seu desempenho.
- **Teste menor (15%)**: permite treinar com mais dados, potencialmente melhorando o ajuste; a avaliação é baseada em menos observações e tende a ser mais sensível à composição do conjunto de teste, especialmente em classes raras.
- Como as partições usam o mesmo seed, a comparação é reproduzível, mas as amostras de teste não têm o mesmo tamanho. Pequenas diferenças não provam superioridade. Para estimativa mais confiável, complemente com validação cruzada estratificada repetida e mantenha um teste final intocado.
- Considere a performance adequada apenas se os erros por classe forem aceitáveis para o contexto. Examine recall das classes críticas, precisão, confusões sistemáticas e estabilidade, e compare com a baseline majoritária.

## Instalação e execução

Requer Python 3.10 ou superior (dependências compatíveis listadas em `requirements.txt`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/analysis.py
```

Para indicar outro local do CSV ou rodar sem abrir janelas de gráficos:

```bash
python src/analysis.py --data /caminho/renewable_energy_production_dataset.csv --no-show
```

No Jupyter, abra `notebooks/challenge_sprint_4.ipynb` a partir da raiz do projeto e execute as células. Se necessário, instale `ipykernel`/`jupyter` no ambiente.

## Artefatos produzidos

| Arquivo | Conteúdo |
|---|---|
| `outputs/correlation_heatmap.png` | Heatmap de correlações |
| `outputs/correlation_matrix.csv` | Matriz numérica completa usada no heatmap |
| `outputs/confusion_matrix_cenário_1.png` | Matriz de confusão — 40% teste |
| `outputs/confusion_matrix_cenário_2.png` | Matriz de confusão — 15% teste |
| `outputs/metrics_comparison.csv` | Comparação numérica de métricas e tamanhos |
| `outputs/classification_reports.txt` | Relatório detalhado por classe para cada cenário |

## Reprodutibilidade e limitações

O seed é fixado em 42. O pipeline previne vazamento ao ajustar imputação, escala e categorias somente em cada conjunto de treino. Categorias inéditas no teste são toleradas pelo encoder. Quando uma partição é pequena demais para conter todas as classes, o código avisa e não estratifica; nesse caso, interprete as métricas por classe com cautela. A regressão logística linear pode não capturar fronteiras complexas; modelos não lineares e validação cruzada são extensões possíveis, sem substituir a análise solicitada.
