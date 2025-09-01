from . import azure
from . import fireworks
from . import gpt
from . import gpt2
from . import gpt3
from . import dummy
from . import seq2seq
from . import google
from . import openai_compatible_models

MODEL_REGISTRY = {
    "azure": azure.AZURECHATGPTLM,
    "gpt": gpt.GPTLM,
    "seq2seq": seq2seq.Seq2SeqLM,
    "fireworks": fireworks.FireworksLM,
    "google": google.GoogleLM,
    "fireworks": openai_compatible_models.FireworksAPI,
    "deepinfra": openai_compatible_models.DeepinfraAPI,
    "deepseek": openai_compatible_models.DeekseekAPI,
    "tgi": openai_compatible_models.TGIAPI,
    "vllm": openai_compatible_models.VLLMAPI,
    "gemini": openai_compatible_models.GeminiAPI,
    "maritalk": openai_compatible_models.MaritalkAPI,
    "chatgpt": openai_compatible_models.OpenaiAPI,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
