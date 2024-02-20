from pydantic import BaseModel


class TrainingArgs(BaseModel):
    output_dir: str = "./training_data"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    optim: str = "paged_adamw_32bit"
    save_steps: int = 0
    logging_steps: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    fp16: bool = False
    bf16: bool = False
    max_grad_norm: float = 0.3
    max_steps: float = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    lr_scheduler_type: str = "cosine"
    report_to: str = "tensorboard"
