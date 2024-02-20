from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ModelConfig(BaseModel):
    pretrained_model_name_or_path: Annotated[
        str, Field(description="Pretrained model path or name")
    ]
    device_map: Annotated[
        str | dict, Field(description="a map that specifies where \
        each submodule should go. To have Accelerate compute the most \
        optimized device_map automatically, set device_map='auto'")
    ]
    use_cache: Annotated[
        bool, Field(description="Whether or not the model should \
        return the last key/values attentions")
    ] = False
    pretraining_tp: Annotated[
        int, Field(description="a value different than 1 will activate \
        the more accurate but slower computation of the linear layers")
    ] = 1
