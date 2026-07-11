import shutil
from pathlib import Path

import kagglehub

DATASET = "marlesson/news-of-the-site-folhauol"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_dataset() -> Path:
    cache_path = Path(kagglehub.dataset_download(DATASET))
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        shutil.copy2(item, RAW_DATA_DIR / item.name)

    return RAW_DATA_DIR
