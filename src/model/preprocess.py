from pathlib import Path

import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

LEMMATIZED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "model"

_nlp = None

# Carregamento do NLP do spaCy
def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
    return _nlp

# Lematização do texto usando o spaCy
def _lemmatize_doc(doc) -> str:
    tokens = [token.lemma_.lower() for token in doc if not (token.is_stop or token.is_punct or token.is_space)]
    return " ".join(tokens)

# Classe sickit-learn para lematização dentro do modelo
class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nlp = _get_nlp()
        X = list(X)
        # Só vale a pena pagar o custo de subir processos paralelos em lotes grandes (treino).
        # Pra poucas linhas (ex: 1 request da API), roda tudo no processo atual.
        n_process = 8 if len(X) > 500 else 1
        return [_lemmatize_doc(doc) for doc in nlp.pipe(X, batch_size=200, n_process=n_process)]

# Lematiza as colunas de texto do dataset, pra poder cachear
def lemmatize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lemmatizer = SpacyLemmatizer()
    df["title"] = lemmatizer.transform(df["title"])
    df["text"] = lemmatizer.transform(df["text"])
    return df


def save_lemmatized_dataset(df: pd.DataFrame, filename: str = "articles_lemmatized.csv") -> Path:
    lemmatized_df = lemmatize_dataset(df)

    LEMMATIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LEMMATIZED_DATA_DIR / filename
    lemmatized_df.to_csv(output_path, index=False)

    return output_path
