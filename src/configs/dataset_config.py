from pydantic import BaseModel, Field
from typing_extensions import Annotated


class DatasetConfig(BaseModel):
    name: Annotated[str, Field(description="dataset name")]
    split: Annotated[str, Field(
        description="dataset split to use for training")] = "train"
