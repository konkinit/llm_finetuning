from fastapi import APIRouter, Body, status
from typing import Annotated

from ..schemas import FinetuningData, FinetuneResponse
from ..tuning.trainer import FineTuner


router = APIRouter(prefix="/finetuning")


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=FinetuneResponse,
    name="finetune_llm",
)
async def create_finetuning(
    finetuning_data: Annotated[FinetuningData, Body(...)],
) -> dict:
    finetuner = FineTuner(finetuning_data)
    finetuner._train()
    finetuner._push_to_HF_hub()
    return FinetuneResponse(
        base_model=finetuning_data.model_config.pretrained_model_name_or_path
    )
