import random
import torch
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
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        return self.batch_size_per_gpu

    @property
    def device(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        return self._device

    def tok_encode(self, string: str):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
