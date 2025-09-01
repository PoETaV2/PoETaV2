import json
import openai
import os
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from lm_eval.base import ModelCategory

def process_chunk(
    chunk_until,
    client,
    engine,
    response_format_obj,
    max_gen_toks=None,
    supports_temperature_stop=True,
):
    chunk, until = chunk_until
    chunk_res = []
    use_completion_tokens = engine in {"o1", "o3", "gpt-5"}
    for context, until_ in chunk:
        is_valid_chat_format = False
        try:
            messages = json.loads(context)
            is_valid_chat_format = (
                isinstance(messages, list) and
                all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages)
            )
        except json.decoder.JSONDecodeError:
            # It's not a valid JSON, continue with is_valid_chat_format = False
            pass

        kwargs = {"model": engine}
        if supports_temperature_stop and not use_completion_tokens:
            kwargs.update({"temperature": 0.0, "stop": []})

        if use_completion_tokens:
            kwargs["max_completion_tokens"] = max_gen_toks
        else:
            kwargs["max_tokens"] = max_gen_toks

        if is_valid_chat_format:
            if (
                messages[0]["role"] == "system"
                and messages[0]["content"] == "You are a helpful assistant."
            ):
                # in the past we removed the system prompt, now we support it,
                # for legacy reasons we still remove the first message if it is the default system prompt
                # but we keep it if it is a custom system prompt
                messages = messages[1:]

            kwargs["messages"] = messages

            
            if response_format_obj:
                response = openai_completion(
                    client=client,
                    is_chat=True, 
                    response_format=response_format_obj,
                    **kwargs
                )
                s = response.choices[0].message.parsed
                # parse obj to json string
                s = s.model_dump_json()
            else:
                response = openai_completion(
                    client=client,
                    is_chat=True, 
                    **kwargs,
                )
                s = response.choices[0].message.content
        else:
            kwargs["prompt"] = context
            if "stop" in kwargs:
                kwargs["stop"] = until

            response = openai_completion(
                client=client,
                is_chat=False,
                **kwargs,
            )
            s = response.choices[0].text

        if s is None:
            s = ""
            print(f"Model returned empty response for context:\n{context}.\nAssuming answer as empty string.")

        chunk_res.append(s)
    return chunk_res


def openai_completion(client, is_chat=False, **kwargs):
    """Query OpenAI API for completion or chat completion.

    Retry with back-off until they respond.
    """
    backoff_time = 3
    max_retry = 5
    n_retry = 0
    
        
    while n_retry < max_retry:
        try:
            is_using_response_format = kwargs.get("response_format")
            if is_chat and not is_using_response_format:
                return client.chat.completions.create(**kwargs)
            elif is_chat and is_using_response_format:
                return client.beta.chat.completions.parse(**kwargs)
            else:
                return client.completions.create(**kwargs)
        except openai.OpenAIError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5
        n_retry += 1
    raise Exception(f"Failed to get a response from the API, tried {max_retry} times")


class OpenaiCompatibleModel(BaseLM):
    MODEL_CATEGORY = ModelCategory.CHAT_MODEL
    SUPPORTS_RESPONSE_FORMAT = True
    REQ_CHUNK_SIZE = 1

    def __init__(
        self,
        engine,
        base_url,
        key_env_var="OPENAI_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None,
        supports_temperature_stop=True
    ):
        """
        Initialize an OpenAI-compatible model.

        Args:
            model_name (str): The name of the model to use.
            base_url (str): The base URL for the API endpoint.
            key_env_var (str, optional): The name of the environment variable containing the API key. Defaults to "OPENAI_API_SECRET_KEY".
            batch_size (int, optional): The batch size for processing requests. Defaults to 1.

        Raises:
            AssertionError: If the specified environment variable for the API key is not found.
        """
        super().__init__()

        assert (
            key_env_var in os.environ
        ), f"Environment variable {key_env_var} not found"

        self.engine = engine
        self.parallel_requests = batch_size
        if engine in {"o1", "o3"}:
            supports_temperature_stop = False
        self.supports_temperature_stop = supports_temperature_stop
        self.base_url = base_url
        self.response_format_obj = response_format_obj
        self.client = openai.OpenAI(
            api_key=os.environ.get(key_env_var),
            base_url=base_url,
        )

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # We assume we are using OpenAI models that support at least 8k tokens (4k prompt + 4k generation)
        return 32_000

    @property
    def max_gen_toks(self):
        return 1000

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        raise NotImplementedError()

    def tok_decode(self, tokens):
        raise NotImplementedError()

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

        process_chunk_partial = functools.partial(
            process_chunk,
            engine=self.engine,
            client=self.client,
            response_format_obj=self.response_format_obj,
            max_gen_toks=self.max_gen_toks,
            supports_temperature_stop=self.supports_temperature_stop,
        )

        chunk_results = [None] * len(re_ord.get_reordered())
        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            futures = []
            for chunk in sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE):
                futures.append(executor.submit(process_chunk_partial, chunk))

            for future in tqdm(as_completed(futures), total=len(futures)):
                chunk_results[futures.index(future)] = future.result()

        res = [item for sublist in chunk_results for item in sublist]

        # partial caching
        for i, (context, until) in enumerate(re_ord.get_reordered()):
            self.cache_hook.add_partial("greedy_until", (context, until), res[i])

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()


class OpenaiAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://api.openai.com/v1",
        key_env_var="OPENAI_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)


class MaritalkAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://chat.maritaca.ai/api",
        key_env_var="MARITALK_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)


class DeekseekAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://api.deepseek.com",
        key_env_var="DEEPSEEK_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)


class TogetherAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://api.together.xyz/v1",
        key_env_var="TOGETHER_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)


class FireworksAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://api.fireworks.ai/inference/v1",
        key_env_var="FIREWORKS_API_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)


class DeepinfraAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://api.deepinfra.com/v1/openai",
        key_env_var="DEEPINFRA_API_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)

class TGIAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="http://localhost:8080/v1",
        key_env_var="TGI_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        os.environ["TGI_API_SECRET_KEY"] = "-"
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj)


class VLLMAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="http://localhost:8000/v1",
        key_env_var="VLLM_API_SECRET_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        os.environ["VLLM_API_SECRET_KEY"] = "-"
        super().__init__(engine, base_url, key_env_var, batch_size, response_format_obj=response_format_obj)


class GeminiAPI(OpenaiCompatibleModel):
    def __init__(
        self,
        engine,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env_var="GEMINI_API_KEY",
        batch_size=1,
        response_format_obj=None
    ):
        super().__init__(
            engine,
            base_url,
            key_env_var,
            batch_size,
            supports_temperature_stop=False,
            response_format_obj=response_format_obj
        )
