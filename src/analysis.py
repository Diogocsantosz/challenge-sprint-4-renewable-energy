"""CHALLENGE SPRINT 4 — classificação da classe de energia renovável.

Coloque renewable_energy_production_dataset.csv em ../data/ e execute:
    python src/analysis.py

Os artefatos são salvos em outputs/. O script não assume os preditores, além
do nome da coluna-alvo Energy_Class, para tolerar variações do arquivo Kaggle.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "renewable_energy_production_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
TARGET = "Energy_Class"
RANDOM_STATE = 42


def _make_one_hot_encoder() -> OneHotEncoder:
    """Compatibilidade com versões do scikit-learn anteriores à 1.2."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - depende da versão instalada
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Carrega e valida o CSV, removendo linhas sem rótulo conhecido."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}\n"
            "Baixe o CSV do Kaggle e salve-o nesse caminho (ou use --data PATH)."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(
            f"A coluna-alvo obrigatória {TARGET!r} não existe. "
            f"Colunas encontradas: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("O arquivo CSV não contém registros.")

    missing_targets = int(df[TARGET].isna().sum())
    if missing_targets:
        warnings.warn(
            f"{missing_targets} linhas sem {TARGET} foram removidas; "
            "não podem ser usadas em aprendizagem supervisionada.",
            stacklevel=2,
        )
        df = df.dropna(subset=[TARGET]).copy()
    if df[TARGET].nunique() < 2:
        raise ValueError("A classificação exige pelo menos duas classes na coluna-alvo.")
    return df.reset_index(drop=True)


def encode_target_and_prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, list[str]]:
    """Codifica y e remove preditores totalmente vazios.

    Categóricas de X permanecem em seu formato original para serem tratadas
    corretamente com one-hot encoding dentro do pipeline, sem vazamento.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    X = df.drop(columns=[TARGET]).copy()
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        warnings.warn(
            f"Preditores totalmente vazios removidos: {all_missing}", stacklevel=2
        )
        X = X.drop(columns=all_missing)
    if X.shape[1] == 0:
        raise ValueError("Não restaram preditores após remover a coluna-alvo.")
    return X, y, encoder, all_missing


def encoded_correlation_matrix(
    X: pd.DataFrame, y: np.ndarray, encoder: LabelEncoder
) -> pd.DataFrame:
    """Matriz Pearson completa após transformar preditores categóricos.

    Colunas categóricas são expandidas em indicadores one-hot; Energy_Class é
    incluída como rótulo numérico LabelEncoder, conforme o enunciado.
    """
    numeric = X.select_dtypes(include=[np.number, "bool"]).copy()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    if categorical_columns:
        categorical = pd.get_dummies(
            X[categorical_columns].astype("string"),
            prefix=categorical_columns,
            dummy_na=True,
            dtype=float,
        )
        numeric = pd.concat([numeric, categorical], axis=1)

    # Datas já tipadas como datetime são representadas por timestamp em ns.
    datetime_columns = X.select_dtypes(include=["datetime", "datetimetz"]).columns
    for column in datetime_columns:
        numeric[column] = X[column].astype("int64").where(X[column].notna(), np.nan)

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric[TARGET] = y
    return numeric.corr(method="pearson", numeric_only=True)


def plot_correlation_matrix(correlation: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plota e salva o heatmap; adapta o tamanho a matrizes largas."""
    n = len(correlation.columns)
    side = min(max(10, n * 0.65), 32)
    fig, ax = plt.subplots(figsize=(side, side * 0.82))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.25 if n < 35 else 0,
        annot=n <= 18,
        fmt=".2f",
        cbar_kws={"label": "Correlação de Pearson"},
        ax=ax,
    )
    ax.set_title("Matriz de correlação — preditores codificados e Energy_Class")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def report_correlations(correlation: pd.DataFrame, threshold: float = 0.85) -> None:
    """Imprime associações com o alvo e pares de alta correlação absoluta."""
    target_corr = correlation[TARGET].drop(labels=[TARGET], errors="ignore").dropna()
    target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
    print("\nCorrelações com Energy_Class codificada (maiores |r| primeiro):")
    print(target_corr.head(20).to_string() if not target_corr.empty else "Sem correlações calculáveis.")

    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    # `dropna` não é mais aceito em algumas versões recentes do pandas.
    pairs = upper.stack()
    pairs = pairs[pairs.abs() >= threshold].sort_values(key=np.abs, ascending=False)
    print(f"\nPares com |correlação| >= {threshold:.2f}:")
    if pairs.empty:
        print("Nenhum par atingiu o limiar.")
    else:
        for (left, right), value in pairs.items():
            print(f"{left} ↔ {right}: {value:.3f}")


def build_model(X: pd.DataFrame) -> Pipeline:
    """Monta regressão logística com imputação, codificação e escala."""
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _split_data(
    X: pd.DataFrame, y: np.ndarray, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Usa divisão estratificada quando o tamanho permite representar as classes."""
    n_classes = len(np.unique(y))
    n_test = int(np.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    counts = pd.Series(y).value_counts()
    can_stratify = (
        counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes
    )
    stratify = y if can_stratify else None
    if not can_stratify:
        warnings.warn(
            f"Divisão test_size={test_size:.2f} não comporta estratificação segura; "
            "será usada divisão aleatória simples. Confira se todas as classes "
            "aparecem nos conjuntos de treino e teste.",
            stacklevel=2,
        )
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def evaluate_scenario(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: np.ndarray,
    test_size: float,
    scenario_name: str,
    output_dir: Path,
    show: bool,
) -> tuple[dict[str, Any], str]:
    """Treina, avalia, salva matriz gráfica e retorna métricas e relatório."""
    X_train, X_test, y_train, y_test = _split_data(X, y, test_size)
    model = build_model(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    labels = np.arange(len(class_names))

    metrics = {
        "cenario": scenario_name,
        "test_size": test_size,
        "n_treino": len(y_train),
        "n_teste": len(y_test),
        "acuracia": accuracy_score(y_test, predictions),
        "acuracia_baseline_majoritaria": accuracy_score(y_test, baseline_predictions),
        "precisao_weighted": precision_score(
            y_test, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "precisao_macro": precision_score(
            y_test, predictions, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, labels=labels, average="macro", zero_division=0
        ),
    }
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=[str(name) for name in class_names],
        zero_division=0,
        digits=4,
    )

    print(f"\n{'=' * 72}\n{scenario_name} — treino {1-test_size:.0%} / teste {test_size:.0%}")
    print(f"Amostras: treino={len(y_train)}, teste={len(y_test)}")
    print(f"Acurácia: {metrics['acuracia']:.4f}")
    print(f"Acurácia baseline majoritária: {metrics['acuracia_baseline_majoritaria']:.4f}")
    print(f"Precisão weighted: {metrics['precisao_weighted']:.4f}")
    print(f"Recall weighted:   {metrics['recall_weighted']:.4f}")
    print(f"Precisão macro:    {metrics['precisao_macro']:.4f}")
    print(f"Recall macro:      {metrics['recall_macro']:.4f}")
    print("\nClassification report:\n" + report)

    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.1), max(5, len(class_names) * 0.9)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False, xticks_rotation=35)
    ax.set_title(f"Matriz de confusão — {scenario_name} ({test_size:.0%} teste)")
    plt.tight_layout()
    safe_name = scenario_name.lower().replace(" ", "_")
    fig.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return metrics, report


def run_analysis(data_path: Path = DEFAULT_DATA_PATH, show: bool = True) -> pd.DataFrame:
    """Executa análise exploratória e os dois experimentos pedidos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    print(f"Dataset: {data_path}\nDimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print("\nTipos e valores ausentes:")
    print(pd.DataFrame({"tipo": df.dtypes.astype(str), "ausentes": df.isna().sum()}).to_string())
    print(f"\nDistribuição de {TARGET}:")
    print(df[TARGET].value_counts(dropna=False).to_string())

    X, y, target_encoder, dropped = encode_target_and_prepare_features(df)
    print("\nCodificação do alvo:")
    print(dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))
    if dropped:
        print(f"Preditores removidos por ausência total: {dropped}")

    correlation = encoded_correlation_matrix(X, y, target_encoder)
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv", encoding="utf-8")
    plot_correlation_matrix(correlation, OUTPUT_DIR, show)
    report_correlations(correlation)

    class_names = target_encoder.classes_
    outcomes: list[dict[str, Any]] = []
    reports: list[str] = []
    for test_size, scenario in [(0.40, "Cenário 1"), (0.15, "Cenário 2")]:
        metrics, report = evaluate_scenario(
            X, y, class_names, test_size, scenario, OUTPUT_DIR, show
        )
        outcomes.append(metrics)
        reports.append(f"{scenario} (teste {test_size:.0%})\n{report}")

    results = pd.DataFrame(outcomes)
    results.to_csv(OUTPUT_DIR / "metrics_comparison.csv", index=False, encoding="utf-8")
    (OUTPUT_DIR / "classification_reports.txt").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )
    print("\nComparação dos cenários:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtefatos salvos em: {OUTPUT_DIR}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Caminho do CSV (padrão: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Salva os gráficos sem abrir janelas (útil em execução headless).",
    )
    args = parser.parse_args()
    run_analysis(args.data, show=not args.no_show)


if __name__ == "__main__":
    main()
