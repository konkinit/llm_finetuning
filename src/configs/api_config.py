from pydantic import BaseModel


class APIConfig(BaseModel):
    title: str = "LLM Finetuning Platform API"
    version: str = "1.0.0"
    description: str = "A FASTApi-based API of LLM Finetuning"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/v1/openapi.json"
    api_prefix: str = "/v1"
    debug: bool = True
