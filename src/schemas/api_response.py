from pydantic import BaseModel


class FinetuneResponse(BaseModel):
    base_model: str
    push_to_hub: str = "successful"
