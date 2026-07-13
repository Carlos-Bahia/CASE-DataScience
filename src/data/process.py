import re
from pathlib import Path

import pandas as pd

MISSING_VALUE = "N/D"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_MULTIPLE_SPACES_RE = re.compile(r"\s{2,}")
_DASH_UNDERSCORE_RE = re.compile(r"[-_]")
_DIGITS_RE = re.compile(r"\d+")

DATASET_FIELDS = ["title", "text", "category", "subcategory", "date", 'link']
DATASET_TEXT_FIELDS = ["title", "text", "category", "subcategory"]


# Remover ruidos HTML, URLs e espaços extras
# Padronizar para minúsculas
def clean_text(text: str) -> str:
    text = _HTML_TAG_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    return text.strip().lower()


# Normaliza rótulos categorias: trocando -/_ por espaço e removendo digítos
def normalize_label(value: str) -> str:
    value = _DASH_UNDERSCORE_RE.sub(" ", value)
    value = _DIGITS_RE.sub("", value)
    return clean_text(value)


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

            # Normaliza categoria e subcategoria
            if (field in ("category", "subcategory")):
                row[field] = normalize_label(str(value)) if not (pd.isna(value) or not str(value).strip()) \
                                                                     else MISSING_VALUE
                
            # Limpa título e texto
            else:
                row[field] = clean_text(str(value)) if not (pd.isna(value) or not str(value).strip()) \
                                                    else MISSING_VALUE

    return row


# Limpeza de dataset inteiro
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["title", "text"]).copy()

    for field in DATASET_TEXT_FIELDS:
        # Normaliza categoria e subcategoria
        if field in ("category", "subcategory"):
            df[field] = df[field].apply(
                lambda value: normalize_label(str(value)) if not (pd.isna(value) or not str(value).strip()) \
                                                            else MISSING_VALUE
            )

        # Limpa título e texto
        else:
            df[field] = df[field].apply(
                lambda value: clean_text(str(value)) if not (pd.isna(value) or not str(value).strip()) \
                                                      else MISSING_VALUE
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
