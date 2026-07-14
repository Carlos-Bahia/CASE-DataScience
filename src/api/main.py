import pandas as pd
from fastapi import FastAPI

from src.api.schemas import NewsInput, NewsPrediction
from src.data.process import clean_row
from src.model.train import NewsClassifierModel

app = FastAPI(title="News Classifier API")

model = NewsClassifierModel.load()


def _predict(items: list[NewsInput]) -> list[NewsPrediction]:
    rows = [clean_row(item.model_dump()) for item in items]
    df = pd.DataFrame(rows)[["title", "text", "subcategory"]]
    categories = model.predict(df)

    return [
        NewsPrediction(**item.model_dump(), category=category)
        for item, category in zip(items, categories)
    ]


@app.post("/predict", response_model=NewsPrediction)
def predict_one(item: NewsInput) -> NewsPrediction:
    return _predict([item])[0]


@app.post("/predict/batch", response_model=list[NewsPrediction])
def predict_batch(items: list[NewsInput]) -> list[NewsPrediction]:
    return _predict(items)
