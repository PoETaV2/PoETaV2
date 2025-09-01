"""
The Unified National Public Competition, also called the Unified National Competition, is an initiative of the Ministry of Management and Innovation in Public Services of the Federal Government of Brazil to centralize and streamline the process of hiring new federal public servants.
"""

import collections
import numpy as np
import re
from lm_eval import utils
import re

from lm_eval.base import rf, MultipleChoicePromptSelectionTask
from lm_eval.metrics import mean
from datasets import load_dataset, load_from_disk
import os
import json
import requests
import random
from lm_eval.base import ModelCategory

_CITATION = """
"""

# first 4 general education questions from ENADE 2021 that does not require image understanding.


class BigBenchJsonTaskBase(MultipleChoicePromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "bigbench/bigbench"
    LOCAL_PATH = "bigbench/X"
    JSON_URL = (
        "https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/X.json"
    )
    DATASET_NAME = None

    KEYS_TO_INDEX = ["input"]
    SEARCHER_K = 10
    shuffle_alternatives = False

    description_as_system_message = False  # when using conversation template, the description is used as system message, instead of being added as a user message

    # task type
    # target_scores_to_alternatives: convert target_scores to alternatives
    # spelled_classification: mode needs to say the correct class, the classes are in the target_scores
    task_type = "target_scores_to_alternatives"

    # can be used to map a class name that is too verbose ( like "not a joke" to another thing like "serious")
    class_mapping = {}

    example_input_prefix = ""
    example_output_prefix = ""

    def handle_multiple_path_download(self):
        assert len(self.LOCAL_PATH) == len(
            self.JSON_URL
        ), "LOCAL_PATH and JSON_URLs must have the same length"

        for local_path, json_url in zip(self.LOCAL_PATH, self.JSON_URL):
            # check if local path exists
            if not os.path.exists(local_path):
                # download json from url
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                response = requests.get(json_url)
                with open(local_path, "w") as f:
                    f.write(response.text)

        # load data from all local paths
        big_bench_task_data = {}
        all_examples = []
        for local_path in self.LOCAL_PATH:
            with open(local_path, "r") as f:
                big_bench_task_data[local_path] = json.load(f)
                all_examples.extend(big_bench_task_data[local_path]["examples"])

        random.shuffle(all_examples)
        big_bench_task_data["examples"] = all_examples

        return big_bench_task_data

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        if isinstance(self.LOCAL_PATH, list):
            self.big_bench_task_data = self.handle_multiple_path_download()
        else:
            # check if local path exists
            if not os.path.exists(self.LOCAL_PATH):
                # download json from url
                os.makedirs(os.path.dirname(self.LOCAL_PATH), exist_ok=True)
                response = requests.get(self.JSON_URL)
                with open(self.LOCAL_PATH, "w") as f:
                    f.write(response.text)

            # load json from local path
            with open(self.LOCAL_PATH, "r") as f:
                self.big_bench_task_data = json.load(f)

        # uncomment to get prefix from json, if it exists
        # self.example_input_prefix = self.big_bench_task_data["example_input_prefix"] if "example_input_prefix" in self.big_bench_task_data else ""
        # self.example_output_prefix = self.big_bench_task_data["example_output_prefix"] if "example_output_prefix" in self.big_bench_task_data else ""

        self.dataset = collections.defaultdict(list)

        examples = self.big_bench_task_data["examples"]

        # add id to each example
        for i, example in enumerate(examples):
            if "id" not in example:
                example["id"] = i

        for example in examples:
            if self.task_type == "target_scores_to_alternatives":
                if len(example["target_scores"]) < 2:
                    continue
            self.dataset["train"].append(self._process_doc(example))

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def test_docs(self):
        return self.dataset["train"]

    def apply_multiple_choice_template(self, input, alternatives):

        letters = [
            "A)",
            "B)",
            "C)",
            "D)",
            "E)",
            "F)",
            "G)",
            "H)",
            "I)",
            "J)",
            "K)",
            "L)",
            "M)",
        ]

        total_alternatives_text = ""
        for letter, alternative_text in zip(letters[: len(alternatives)], alternatives):
            total_alternatives_text += f"{letter} {alternative_text}\n"

        return f"{self.example_input_prefix}{input}\n{total_alternatives_text}"

    def _process_doc(self, doc):

        assert "input" in doc, "input is not in doc"

        if self.task_type == "target_scores_to_alternatives":
            assert "target_scores" in doc, "target_scores is not in doc"

            alternatives = []
            correct_index = None

            target_scores = list(doc["target_scores"].items())
            if self.shuffle_alternatives:
                random.shuffle(target_scores)
            letters = [
                "A)",
                "B)",
                "C)",
                "D)",
                "E)",
                "F)",
                "G)",
                "H)",
                "I)",
                "J)",
                "K)",
                "L)",
                "M)",
            ]
            for alternative_text, value in target_scores:
                correct = alternative_text if value == 1 else None
                alternative_text = (
                    self.class_mapping[alternative_text]
                    if alternative_text in self.class_mapping
                    else alternative_text
                )
                alternatives.append(alternative_text)
                if correct is not None:
                    correct_index = len(alternatives) - 1
            assert correct_index is not None, "correct_index is not in doc"

            return {
                "query": self.apply_multiple_choice_template(
                    doc["input"], alternatives
                ),
                "choices": alternatives,
                "gold": letters[correct_index],
                "correct_index": correct_index,
                "id": doc["id"],
            }
        elif self.task_type == "spelled_classification":

            possible_classes = doc["target_scores"].keys()
            # apply class mapping if it exists
            possible_classes = [
                self.class_mapping[c] if c in self.class_mapping else c
                for c in possible_classes
            ]

            # get correct class, apply class mapping if it exists
            correct_class = list(doc["target_scores"].keys())[
                list(doc["target_scores"].values()).index(1)
            ]
            correct_class = (
                self.class_mapping[correct_class]
                if correct_class in self.class_mapping
                else correct_class
            )

            prompt = f"{self.example_input_prefix}{doc['input']}\n{self.example_output_prefix}"

            return {
                "query": prompt,
                "gold": correct_class,
                "possible_classes": possible_classes,
                "id": doc["id"],
            }

    def doc_to_text(self, doc):
        if (
            self.task_type == "target_scores_to_alternatives"
            or self.task_type == "spelled_classification"
        ):
            return doc["query"]
        else:
            return doc["input"]

    def fewshot_examples(self, k, rnd, prompt_mode, doc):
        if prompt_mode == "manual":
            assert k <= len(self.manual_examples), (
                f"The number of manual_examples is not enough to satisfy "
                f"num_fewshot={k}. Please, include more examples."
            )
            return self.manual_examples[:k]
        else:
            return super().fewshot_examples(k, rnd, prompt_mode, doc)

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }

    @utils.positional_deprecated
    def fewshot_context(
        self,
        doc,
        num_fewshot,
        prompt_mode=None,
        provide_description=None,
        rnd=None,
        description=None,
        conversation_template=None,
        **kwargs,
    ):

        return super().fewshot_context(
            doc=doc,
            num_fewshot=num_fewshot,
            prompt_mode=prompt_mode,
            provide_description=provide_description,
            rnd=rnd,
            description=description,
            conversation_template=conversation_template,
            **kwargs,
        )


class BigBenchTaskBaseGreedy(BigBenchJsonTaskBase):

    def doc_to_text(self, doc):
        text_to_return = ""
        if (
            self.task_type == "target_scores_to_alternatives"
            or self.task_type == "spelled_classification"
        ):
            text_to_return = doc["query"]
        else:
            text_to_return = doc["input"]

        if self.inference_model_category == ModelCategory.COMPLETION_MODEL:
            text_to_return = text_to_return + "Resposta:"

        return text_to_return

    def doc_to_target(self, doc):
        return " " + doc["gold"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        continuation = rf.greedy_until(ctx, ["\n"])
        return continuation

    def process_results(self, doc, results):

        pred = results[0].strip()

        if self.task_type == "target_scores_to_alternatives":

            correct_letter = doc["gold"]

            full_pred = pred

            # search for LETTER) in pred
            letter_match = re.search(r"[A-Z]\)", pred)
            if letter_match:
                regex_pred = letter_match.group(0)
            else:
                regex_pred = ""
            if pred.lower().strip() == correct_letter.lower().strip():
                acc = 1
            elif regex_pred.lower().strip() == correct_letter.lower().strip():
                acc = 1
            else:
                acc = 0

            debug_info = {
                "pred": pred,
                "regex_pred": regex_pred,
                "correct_letter": correct_letter,
                "doc_id": doc["id"],
            }
            return {
                "acc": acc,
                "debug_info": debug_info,
            }
        elif self.task_type == "spelled_classification":
            if pred.strip() == doc["gold"].strip():
                acc = 1
            else:
                acc = 0

            valid_classes = self.possible_classes
            normalized_classes = [c.strip().lower() for c in valid_classes]
            regex_string = "|".join(normalized_classes)
            regex_pred = re.search(regex_string, pred.strip().lower())
            if regex_pred:
                regex_pred = regex_pred.group(0)
            else:
                regex_pred = ""

            if regex_pred.strip().lower() == doc["gold"].strip().lower():
                acc = 1
            else:
                acc = 0

            debug_info = {
                "pred": pred,
                "regex_pred": regex_pred,
                "correct_class": doc["gold"],
                "doc_id": doc["id"],
            }
            return {
                "acc": acc,
                "debug_info": debug_info,
            }
