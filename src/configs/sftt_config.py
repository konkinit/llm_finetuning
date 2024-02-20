from pydantic import BaseModel


class SFTTrainerConfig(BaseModel):
    dataset_text_field: str = "text"
    packing: bool = False
