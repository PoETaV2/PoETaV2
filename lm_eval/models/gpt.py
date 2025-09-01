import os
import torch
import transformers
import peft
from lm_eval.base import BaseLM
from lm_eval.utils import stop_sequences_criteria


class GPTLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        adapter=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        dtype=None,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)
        if dtype:
            assert dtype in ["fp32", "fp16", "bf16", "int8", "int4"]
        if device:
            if isinstance(device, int) or device.isnumeric():
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

        revision = revision + ("/" + subfolder if subfolder is not None else "")

        model_kwargs = {}
        model_kwargs['device_map'] = 'auto'

        if dtype == "fp32":
            model_kwargs['torch_dtype'] = torch.float32
        elif dtype == "fp16":
            model_kwargs['torch_dtype'] = torch.float16
        elif dtype == "bf16":
            model_kwargs['torch_dtype'] = torch.bfloat16
        elif dtype == "int8":
            model_kwargs['load_in_8bit'] = True
        elif dtype == "int4":
            model_kwargs['load_in_4bit'] = True

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=True,
            use_fast=True)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **model_kwargs)

        if adapter: 
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                adapter,
                revision=revision,
                low_cpu_mem_usage=True,
                **model_kwargs)

        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = batch_size

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return getattr(self.model.config, "max_position_embeddings")
        except AttributeError:
            return 4096

    @property
    def max_gen_toks(self):
        return 2048

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, stop_sequences):

        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop_sequences, context.shape[1], context.shape[0]
        )

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )
