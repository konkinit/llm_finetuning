from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer
from src.configs import (
    SFTTrainerConfig, ModelConfig, DatasetConfig,
    TokenizerConfig, LoRAConfig, TrainingArgs, BnBConfig
)


class FineTuner(SFTTrainer):
    def __init__(
        self, model_config: ModelConfig, dataset_config: DatasetConfig,
        bnb_config: BnBConfig, sftt_configs: SFTTrainerConfig,
        tokenizer_config: TokenizerConfig, lora_config: LoRAConfig,
        training_args: TrainingArgs
    ):
        dataset = load_dataset(
            dataset_config.name, split=dataset_config.split)

        bnb_config = BitsAndBytesConfig(
            **bnb_config.model_dump()
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_config.pretrained_model_name_or_path,
            quantization_config=bnb_config,
            device_map=model_config.device_map
        )
        model.config.use_cache = model_config.use_cache
        model.config.pretraining_tp = model_config.pretraining_tp

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.pretrained_model_name_or_path,
            trust_remote_code=tokenizer_config.trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = tokenizer_config.padding_side

        peft_config = LoraConfig(**lora_config.model_dump())

        training_args = TrainingArguments(**training_args.model_dump())

        super().__init__(
            model=model, train_dataset=dataset, peft_config=peft_config,
            tokenizer=tokenizer, args=training_args, **sftt_configs.model_dump()
        )

    def _train(self):
        self.train()

    def _save(self):
        self.model.save_pretrained()
