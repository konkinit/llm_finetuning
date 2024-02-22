from pydantic import BaseModel, Field
from typing_extensions import Annotated


class BnBConfig(BaseModel):
    load_in_4bit: Annotated[
        bool, Field(description="enable 4-bit quantization")
    ] = True
    bnb_4bit_quant_type: Annotated[
        str, Field(description="quantization data type")
    ] = "nf4"
    bnb_4bit_compute_dtype: Annotated[
        str, Field(description="Compute dtype for 4-bit base model")
    ] = "float16"
    bnb_4bit_use_double_quant: Annotated[
        bool, Field(description="enable double quantization")
    ] = False
