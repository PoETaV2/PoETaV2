import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
from lm_eval.base import BaseLM
from lm_eval import utils
import os
from tqdm import tqdm
import time


def convert_messages(messages):
    """Converts API messages from OpenAI to Google GenAI format."""
    converted_messages = []
    for message in messages:
        if message["role"] == "user":
            converted_messages.append({"role": "user", "parts": [message["content"]]})
        else:
            converted_messages.append({"role": "model", "parts": [message["content"]]})
    return converted_messages


class GoogleLM(BaseLM):
    REQ_CHUNK_SIZE = 1

    def __init__(self, engine, truncate=False):
        """

        :param engine: str
            Google GenAI API engine (e.g. gemini-pro)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        # Read from environment variable GOOGLE_API_KEY
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

        self.model = genai.GenerativeModel(
            model_name=engine,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config=genai.GenerationConfig(
                candidate_count=None,
                stop_sequences=None,
                max_output_tokens=self.max_gen_toks,
                temperature=0.0,
                top_p=1.0,
                top_k=None,
            )
        )

    @property
    def eot_token_id(self):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    @property
    def max_length(self):
        return 8192

    @property
    def max_gen_toks(self):
        return 256

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

    def google_completion(self, messages):
        """Query Google GenAI API for completion.

        Retry with back-off until they respond.
        """
        backoff_time = 3
        max_retry = 20
        n_retry = 0
        while n_retry < max_retry:
            try:
                return self.model.generate_content(messages)
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(backoff_time)
                backoff_time *= 1.5
            n_retry += 1

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # ChatGPT does not suppport max_tokens=0 and does not return logprobs
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
            inps = []
            for context, _ in chunk:
                try:
                    messages = json.loads(context)
                    # gemini doesn't use system messages, so we can ignore them
                    if messages[0]["role"] == "system":
                        messages = messages[1:]
                    # convert to gemini format
                    messages = convert_messages(messages)
                except json.decoder.JSONDecodeError:
                    # If context is not a valid JSON string, pass it as is
                    messages = context
                inps.append(messages)

            response = self.google_completion(inps[0])

            for resp, (context, until_) in zip(response.candidates, chunk):
                s = resp.content.parts[0].text

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
