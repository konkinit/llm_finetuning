import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer
from ..schemas.finetuning import FinetuningData
from ..configs.push2hub_config import Push2HubConfig


load_dotenv()


class FineTuner(SFTTrainer):
    def __init__(
        self, finetuning_data: FinetuningData,
    ):
        dataset = load_dataset(
            finetuning_data.dataset_config.name,
            split=finetuning_data.dataset_config.split)

        bnb_config = BitsAndBytesConfig(
            **finetuning_data.bnb_config.model_dump()
        )

        model = AutoModelForCausalLM.from_pretrained(
            finetuning_data.llmodel_config.pretrained_model_name_or_path,
            quantization_config=bnb_config,
            device_map=finetuning_data.llmodel_config.device_map
        )
        model.config.use_cache = finetuning_data.llmodel_config.use_cache
        model.config.pretraining_tp = finetuning_data.llmodel_config.pretraining_tp

        tokenizer = AutoTokenizer.from_pretrained(
            finetuning_data.llmodel_config.pretrained_model_name_or_path,
            trust_remote_code=finetuning_data.tokenizer_config.trust_remote_code)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = finetuning_data.tokenizer_config.padding_side

        peft_config = LoraConfig(**finetuning_data.lora_config.model_dump())

        training_args = TrainingArguments(**finetuning_data.training_args.model_dump())

        super().__init__(
            model=model, train_dataset=dataset, peft_config=peft_config,
            tokenizer=tokenizer, args=training_args,
            **finetuning_data.sfttrainer_config.model_dump()
        )

        self.push2hub_config: Push2HubConfig = finetuning_data.push2hub_config

    def _train(self) -> None:
        self.train()

    def _save(self) -> None:
        self.model.save_pretrained("./results/finetuned_models")

    def _push_to_HF_hub(self) -> None:
        """Push the finetuned model to Hugging Face Hub

        Args:
            push2hub_config (Push2HubConfig): push config data
        """
        login(token=os.getenv("HF_WRITE_TOKEN"))
        self.model.push_to_hub(
            repo_id=f"konkinit/{self.push2hub_config.finetuned_model_name}",
            commit_message=self.push2hub_config.commit_message
        )
