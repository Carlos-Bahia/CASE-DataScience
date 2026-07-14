from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.model.preprocess import lemmatize_dataset

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "model"


class NewsClassifierModel:
    MIN_CATEGORY_EXAMPLES = 800

    def __init__(self, min_category_examples: int = MIN_CATEGORY_EXAMPLES):
        self.min_category_examples = min_category_examples
        self.pipeline = self._build_pipeline()
        self.metrics = None
        self.y_test = None
        self.y_pred = None

    @staticmethod
    def _build_pipeline() -> Pipeline:
        column_transformer = ColumnTransformer(
                                transformers=[
                                    ("title", TfidfVectorizer(max_features=20000), "title"),
                                    ("text", TfidfVectorizer(max_features=20000), "text"),
                                    ("subcategory", OneHotEncoder(handle_unknown="ignore"), ["subcategory"]),
                                ],
                                transformer_weights={"title": 2.0, "text": 1.0, "subcategory": 0.5},
                            )

        return Pipeline([
            ("features", column_transformer),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    # Filtrar classes muito raras para o treinamento do modelo
    def filter_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df["category"].value_counts()
        valid_categories = counts[counts >= self.min_category_examples].index
        return df[df["category"].isin(valid_categories)]

    def train(self, df: pd.DataFrame, already_lemmatized: bool = False) -> dict:

        # Pre-processa os dados
        df = self.filter_rare_categories(df)

        if not already_lemmatized:
            df = lemmatize_dataset(df)

        df = df.copy()
        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")

        X = df[["title", "text", "subcategory"]]
        y = df["category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42
        )

        # Treina e valida o modelo
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        # Salva Métricas do Modelo
        self.y_test = y_test
        self.y_pred = y_pred

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "classification_report": classification_report(y_test, y_pred),
        }

        return self.metrics

    # Faz a predição de novas amostras
    def predict(self, X):
        X = lemmatize_dataset(X)
        X = X.copy()
        X["title"] = X["title"].fillna("")
        X["text"] = X["text"].fillna("")
        return self.pipeline.predict(X)

    # Salva o modelo treinado para ser carregado posteriormente
    def save(self, filename: str = "class_model.joblib") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MODEL_DIR / filename
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, filename: str = "class_model.joblib") -> "NewsClassifierModel":
        return joblib.load(MODEL_DIR / filename)
