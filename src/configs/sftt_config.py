from pydantic import BaseModel, Field
from typing_extensions import Annotated


class SFTTrainerConfig(BaseModel):
    dataset_text_field: Annotated[
        str, Field(description="dataset text field")
    ] = "text"
    packing: Annotated[
        bool, Field(description="used to pack the sequences of the dataset.")
    ] = False
