from . import gpt2
from . import gpt3
from . import textsynth
from . import dummy
from . import flan_t5

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    "flant5": flan_t5.FlanT5,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
