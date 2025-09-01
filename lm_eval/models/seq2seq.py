import os
import torch
import transformers
from tqdm import tqdm
import torch.nn.functional as F
from lm_eval import utils
from lm_eval.base import BaseLM


class Seq2SeqLM(BaseLM):
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
            assert dtype in ["fp32", "fp16", "int8"]
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

        if dtype == "fp32":
            model_kwargs['torch_dtype'] = torch.float32

        elif dtype == "fp16":
            model_kwargs['torch_dtype'] = torch.float16
        
        elif dtype == "int8":
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = 'auto'

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained,
            revision=revision,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            max_length=2048,
        )

        if adapter: 
            # Load adapter
            try:
                adapter_name = self.model.load_adapter(f'{adapter}/clm')
            except RuntimeError:
                # Work around: padding the lm-head just once in case of vocab sizes mismatching
                head = torch.load(f'{adapter}/clm/pytorch_model_head.bin')
                pad_right = self.model.lm_head.weight.shape[0] - head['lm_head.weight'].shape[0]
                head['lm_head.weight'] = torch.nn.functional.pad(head['lm_head.weight'], (0, 0, 0, pad_right))
                torch.save(head, f'{adapter}/clm/pytorch_model_head.bin')

                adapter_name = self.model.load_adapter(f'{adapter}/clm')

            self.model.set_active_adapters(adapter_name)
            print(f'The adapter {adapter_name} is active')

            # Load embedding
            if os.path.exists(f'{adapter}/embeddings'):
                self.model.load_embeddings(f'{adapter}/embeddings', 'gptimbau_embeddings')
                print(f'The embedding {self.model.active_embeddings} is active')

            # Overwrite tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(adapter)

        if dtype != "int8":
            self.model = self.model.to(device)

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
        return 2048

    @property
    def max_gen_toks(self):
        return 150

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=True, max_length=self.max_length)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_generate(self, context, max_length, eos_token_id):
        output = self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            do_sample=False
        )
        return torch.cat([context, output], dim=1)

    def _model_call(self, inps):
        raise NotImplementedError('This is not implemented for encoder-decoder models')
