import json
import os
from lm_eval.base import BaseLM
from lm_eval import utils
import requests
from tqdm import tqdm
import time


def fireworks_completion(**kwargs):
    """Query Fireworks API for completion.

    Retry with back-off until they respond.
    """
    api_key = os.environ["FIREWORKS_API_SECRET_KEY"]
    backoff_time = 3
    max_retry = 20
    n_retry = 0
    while n_retry < max_retry:
        try:
            request = requests.post(
                f"https://api.fireworks.ai/inference/v1/chat/completions",
                json=kwargs,
                headers={
                    "Authorization": f"Bearer {api_key}"
                }
            )
            if not request.ok:
                raise ValueError(request.text)
            return request.json()
        except Exception as e:
            import traceback
            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5
        n_retry += 1


class FireworksLM(BaseLM):
    REQ_CHUNK_SIZE = 1

    def __init__(self, engine, truncate=False):
        """

        :param engine: str
            MariTalk API engine (e.g. Maritalk)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        self.engine = engine

    @property
    def eot_token_id(self):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    @property
    def max_length(self):
        return 32768

    @property
    def max_gen_toks(self):
        return 4096

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
    
    def tok_decode(self, tokens):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            return len(x[0]), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            for context, until_ in chunk:
                try:
                    messages=json.loads(context)
                except json.decoder.JSONDecodeError:
                    messages=[{"role": "user", "content": context}]
             
                response = fireworks_completion(
                    model=self.engine,
                    messages=messages,
                    max_tokens=self.max_gen_toks,
                    temperature=0.,
                )
                
                s = response["choices"][0]["message"]["content"].strip()

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until_), s)
                
                res.append(s)
        
        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
