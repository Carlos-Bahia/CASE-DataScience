from typing import Optional

from pydantic import BaseModel, model_validator

class NewsInput(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None
    subcategory: Optional[str] = None
    date: Optional[str] = None
    link: Optional[str] = None

    @model_validator(mode="after")
    def at_least_one_field(self):
        if not (self.title or self.text or self.subcategory):
            raise ValueError("Informe ao menos um dos campos: title, text ou subcategory")
        return self


class NewsPrediction(NewsInput):
    category: str
