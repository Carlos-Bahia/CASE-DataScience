from pathlib import Path

import pandas as pd

from src.data.process import process_and_save
from src.data.reader import fetch_dataset

PROCESSED_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "articles.csv"


def main() -> None:
    if PROCESSED_DATA_PATH.exists():
        print(f"Base processada já existe em {PROCESSED_DATA_PATH}, pulando geração.")
        return

    raw_data_dir = fetch_dataset()
    csv_path = next(raw_data_dir.glob("*.csv"))

    df = pd.read_csv(csv_path)
    process_and_save(df, csv_path.name)


if __name__ == "__main__":
    main()
