import re
from pathlib import Path

import pandas as pd

MISSING_VALUE = "N/D"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_MULTIPLE_SPACES_RE = re.compile(r"\s{2,}")

DATASET_FIELDS = ["title", "text", "category", "subcategory", "date", 'link']
DATASET_TEXT_FIELDS = ["title", "text", "category", "subcategory"]


# Remover ruidos HTML, URLs e espaços extras
# Padronizar para minúsculas
def clean_text(text: str) -> str:
    text = _HTML_TAG_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    return text.strip().lower()


# Limpeza de 1 linha
def clean_row(row: dict) -> dict:
    row = dict(row)

    for field in DATASET_FIELDS:
        # Garante que todos os campos existam na linha
        if field not in row:
            row[field] = None

        # Realiza a limpeza dos campos de texto
        if field in DATASET_TEXT_FIELDS:
            value = row.get(field)
            row[field] = clean_text(str(value)) if pd.isna(value) or not str(value).strip() \
                                                else value

    return row


# Limpeza de dataset inteiro
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["title", "text"]).copy()

    for field in DATASET_TEXT_FIELDS:
        df[field] = df[field].apply(
            lambda value: MISSING_VALUE if pd.isna(value) or not str(value).strip() else clean_text(str(value))
        )

    return df


# Processa o dataframe e salva em data/processed com o mesmo nome do arquivo original
def process_and_save(df: pd.DataFrame, filename: str) -> Path:
    cleaned_df = clean_dataset(df)

    dir = Path(__file__).resolve().parents[2] / "data" / "processed"

    dir.mkdir(parents=True, exist_ok=True)
    output_path = dir / filename
    cleaned_df.to_csv(output_path, index=False)

    return output_path
