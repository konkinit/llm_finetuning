from pydantic import BaseModel


class LoRAConfig(BaseModel):
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    r: int = 16
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
