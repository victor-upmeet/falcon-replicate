from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    text: str


class Predictor(BasePredictor):
    def __init__(self):
        self.tokenizer = None
        self.model = "tiiuae/falcon-7b-instruct"

    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def predict(
        self,
        prompt: str
    ) -> ModelOutput:
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        sequences = pipeline(
            prompt,
            max_length=200000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = ""

        for seq in sequences:
            generated_text += seq['generated_text']

        return ModelOutput(
            text=generated_text
        )

