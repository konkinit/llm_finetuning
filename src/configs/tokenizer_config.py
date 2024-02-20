from pydantic import BaseModel, Field
from typing_extensions import Annotated


class TokenizerConfig(BaseModel):
    trust_remote_code: Annotated[
        bool, Field(description="flag to indicate if remote code \
        should be trusted")
    ] = True
    padding_side: Annotated[
        str, Field(description="padding side.")
    ] = "right"
