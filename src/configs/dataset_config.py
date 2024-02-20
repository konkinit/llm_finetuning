from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    split: str = "train"
