from pydantic import BaseModel, Field
from typing_extensions import Annotated


class TrainingArgs(BaseModel):
    output_dir: Annotated[
        str, Field(description="The output directory where the model \
        predictions and checkpoints will be written.")
    ] = "./finetuning_results/training_data"
    gradient_accumulation_steps: Annotated[
        int, Field(description=" Number of updates steps to accumulate \
        the gradients for, before performing a backward/update pass.")
    ] = 1
    learning_rate: Annotated[
        float, Field(description="The initial learning rate for AdamW optimizer.")
    ] = 2e-4
    weight_decay: Annotated[
        float, Field(description="The weight decay to apply (if not zero) \
        to all layers except all bias and LayerNorm weights in AdamW optimizer.")
    ] = 0.001
    lr_scheduler_type: Annotated[
        str, Field(description="The scheduler type to use.")
    ] = "cosine"
    num_train_epochs: Annotated[
        float, Field(description="Total number of training epochs to perform ")
    ] = 1.0
    per_device_train_batch_size: Annotated[
        int, Field(description="The batch size per GPU/XPU/TPU/MPS/NPU \
        core/CPU for training.")
    ] = 4
    optim: Annotated[
        str, Field(description="The optimizer to use")
    ] = "adafactor"
    save_steps: Annotated[
        float | int, Field(description="Number of updates steps before two \
        checkpoint saves")
    ] = 0.25
    logging_steps: Annotated[
        float | int, Field(description="Number of update steps between two logs")
    ] = 0.25
    fp16: Annotated[
        bool, Field(description="enable mixed precision training")
    ] = True
    bf16: bool = False
    max_grad_norm: Annotated[
        float, Field(description="Maximum gradient norm (for gradient clipping)")
    ] = 0.3
    max_steps: Annotated[
        int, Field(description="f set to a positive number, the total number \
        of training steps to perform. Overrides num_train_epochs")
    ] = -1
    warmup_ratio: Annotated[
        float, Field(description="Ratio of total training steps used for a \
        linear warmup from 0 to learning_rate.")
    ] = 0.03
    group_by_length: Annotated[
        bool, Field(description="Whether or not to group together samples of \
        roughly the same length in the training dataset (to minimize padding \
        applied and be more efficient)")
    ] = True
    report_to: Annotated[
        str, Field(
            description="he list of integrations to report the results and logs to.",
            json_schema_extra={"example": ["all", "tensorboard", "mlflow"]})
    ] = "all"
