import random
import torch
import sys
from lm_eval.base import BaseLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class FlanT5(BaseLM):
    def __init__(self, device, batch_size=1):
        """
        :param engine: str
            TextSynth API engine (e.g. `gptj_6B`)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(
            self._device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        self.batch_size_per_gpu = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # According to this github issue, T5 models are not limited to max_position_embeddings, so they deleted the parameter
        # Link: https://github.com/huggingface/transformers/issues/8047
        # We return a large number for that reason
        return sys.maxsize

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
