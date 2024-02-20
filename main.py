from fastapi import FastAPI, status

from src.api_routes.finetuning_route import router as finetune_route
from src.configs.api_config import APIConfig


fastapi_config = APIConfig().model_dump(
    exclude=["version", "redoc_url", "api_prefix"]
)


app = FastAPI(**fastapi_config)


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict:
    """API root message

    Returns:
        dict: message
    """
    return {"Hello": "Welcome to Konkinit's LLM Finetuning Platform"}


app.include_router(
    finetune_route, prefix=APIConfig().api_prefix, tags=["LLM_Finetuning"]
)
