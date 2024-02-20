from pydantic import BaseModel, Field
from typing_extensions import Annotated


class Push2HubConfig(BaseModel):
    finetuned_model_name: Annotated[
        str, Field(description="finetuned model name")]
    commit_message: Annotated[str, Field(
        description="commit message")]
