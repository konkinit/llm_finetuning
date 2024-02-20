from pydantic import BaseModel


class TokenizerConfig(BaseModel):
    trust_remote_code: bool = True
    padding_side: str = "right"
