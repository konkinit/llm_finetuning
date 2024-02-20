from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str
    device_map: str | dict
    use_cache: Annotated[
        bool, Field(description="Whether or not the model should \
        return the last key/values attentions")
    ] = False
    pretraining_tp: Annotated[
        int, Field(description="a value different than 1 will activate \
        the more accurate but slower computation of the linear layers")
    ] = 1
