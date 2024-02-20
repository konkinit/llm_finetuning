from pydantic import BaseModel
from ..configs.bnb_config import BnBConfig
from ..configs.dataset_config import DatasetConfig
from ..configs.lora_config import LoRAConfig
from ..configs.model_config import ModelConfig
from ..configs.sftt_config import SFTTrainerConfig
from ..configs.tokenizer_config import TokenizerConfig
from ..configs.training_args import TrainingArgs
from ..configs.push2hub_config import Push2HubConfig


class FinetuningData(BaseModel):
    bnb_config: BnBConfig
    dataset_config: DatasetConfig
    lora_config: LoRAConfig
    llmodel_config: ModelConfig
    sfttrainer_config: SFTTrainerConfig
    tokenizer_config: TokenizerConfig
    training_args: TrainingArgs
    push2hub_config: Push2HubConfig
