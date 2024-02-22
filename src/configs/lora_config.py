from pydantic import BaseModel, Field
from typing_extensions import Annotated


class LoRAConfig(BaseModel):
    lora_alpha: Annotated[
        int, Field(
            description="The alpha parameter for Lora scaling. Alpha \
        scales the learned weights. Generally advises fixing Alpha—often \
        at 16—rather than treating it as a tunable hyperparameter")
    ] = 16
    lora_dropout: Annotated[float, Field(
        description="The dropout probability for Lora layers")
    ] = 0.1
    r: Annotated[
        int, Field(description="Lora attention dimension")
    ] = 32
    bias: Annotated[str, Field(
        description="Bias type for LoRA.",
        json_schema_extra={"example": ["none", "all", "‘lora_only’"]}
    )] = "none"
    task_type: Annotated[str, Field(
        description="Task type",
    )] = "CAUSAL_LM"
