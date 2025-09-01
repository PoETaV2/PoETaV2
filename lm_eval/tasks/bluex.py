"""
BLUEX is a dataset containing multiple choice questions extracted from the entrance exam
of two of the major Brazilian universities: University of São Paulo (USP) and University of Campinas (UNICAMP).
"""
import collections
import json
import numpy as np
import os
import re
from zipfile import ZipFile
from pathlib import Path
import subprocess
from datasets import load_dataset

from lm_eval.base import rf, MultipleChoicePromptSelectionTask
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{almeida2023bluex,
  title={Bluex: A benchmark based on Brazilian leading universities entrance exams},
  author={Almeida, Thales Sales and Laitz, Thiago and Bon{\'a}s, Giovana K and Nogueira, Rodrigo},
  booktitle={Brazilian Conference on Intelligent Systems},
  pages={337--347},
  year={2023},
  organization={Springer}
}
"""


PATTERNS_REPLACES = [
    (r"\s*\n+\s*", r" "),  # changing \n to space
    (r"(\s)\1+", r" "),  # changing \n to space
    (r"^\s+", r""),
]


apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)


manual_examples = [
    {
        "query": "Enunciado: O conto “O espelho”, de Machado de Assis, apresenta o esboço de uma teoria sobre a alma humana. A tese apresentada defende a existência de duas almas (interior e exterior), que completam o homem. Contudo, o narrador faz uma distinção entre as almas que mudam de natureza e estado e aquelas que são enérgicas. Escolha a alternativa que ilustra, no conto, a mutabilidade da alma exterior.\nAlternativas:\nA. A liderança é a força sem a qual o poder político de Oliver Cromwell se extingue.\nB. A patente é a marca de distinção, sem a qual Jacobina se extingue.\nC. Os versos de Luís de Camões são uma declaração de amor à pátria, pela qual o poeta se dispõe a morrer.\nD. As moedas de ouro são o valor visível sem o qual Shylock prefere morrer.\nResposta:",
        "choices": [
            "A. A liderança é a força sem a qual o poder político de Oliver Cromwell se extingue.",
            "B. A patente é a marca de distinção, sem a qual Jacobina se extingue.",
            "C. Os versos de Luís de Camões são uma declaração de amor à pátria, pela qual o poeta se dispõe a morrer.",
            "D. As moedas de ouro são o valor visível sem o qual Shylock prefere morrer.",
        ],
        "gold": 1,
        "id": "UNICAMP_2021_4",
        "exam": "2021_2",
        "BK": True,
        "contents": "O conto “O espelho”, de Machado de Assis, apresenta o esboço de uma teoria sobre a alma humana. A tese apresentada defende a existência de duas almas (interior e exterior), que completam o homem. Contudo, o narrador faz uma distinção entre as almas que mudam de natureza e estado e aquelas que são enérgicas. Escolha a alternativa que ilustra, no conto, a mutabilidade da alma exterior.",
        "subject": ["portuguese"],
    },
    {
        "query": "Enunciado: Lélia Gonzalez (1935-1994) teve um papel pioneiro na criação de uma teoria do feminismo negro brasileiro. O momento mais intenso de sua militância ocorreu durante a Ditadura Militar (1964-1985), que coibiu a organização política da sociedade civil. A Lei de Segurança Nacional, de setembro de 1967, estabelecia que era crime “incitar publicamente ao ódio ou à discriminação racial”. O que, na verdade, poderia ser usado contra o movimento negro, uma vez que denunciar o racismo e expor o mito da democracia racial poderia ser considerado uma ameaça à ordem social, um estímulo ao antagonismo e uma incitação ao preconceito. (Adaptado de Raquel Barreto, “Memória - Lélia Gonzalez”. Revista Cult 247. São Paulo, julho, 2019. Disponível em https://revistacult.uol.com.br/home/ lelia- gonzalez-perfil/. Acessado em 01/05/2020.) A partir do excerto sobre Lélia Gonzalez e seu contexto histórico, assinale a alternativa correta.\nAlternativas:\nA. A Ditadura Militar perseguiu o feminismo negro no Brasil por ele pregar a supremacia das mulheres negras.\nB. A Ditadura Militar criou mecanismos para recolher denúncias contra a discriminação e combater o racismo estrutural no país.\nC. A Lei de Segurança Nacional criou instrumentos jurídicos que possibilitavam a criminalização de denúncias contra o racismo.\nD. A Lei de Segurança Nacional possibilitou a harmonia das relações étnico-raciais e a igualdade de gênero no Brasil.\nResposta:",
        "choices": [
            "A. A Ditadura Militar perseguiu o feminismo negro no Brasil por ele pregar a supremacia das mulheres negras.",
            "B. A Ditadura Militar criou mecanismos para recolher denúncias contra a discriminação e combater o racismo estrutural no país.",
            "C. A Lei de Segurança Nacional criou instrumentos jurídicos que possibilitavam a criminalização de denúncias contra o racismo.",
            "D. A Lei de Segurança Nacional possibilitou a harmonia das relações étnico-raciais e a igualdade de gênero no Brasil.",
        ],
        "gold": 2,
        "id": "UNICAMP_2021_63",
        "exam": "2021_2",
        "BK": True,
        "contents": "Lélia Gonzalez (1935-1994) teve um papel pioneiro na criação de uma teoria do feminismo negro brasileiro. O momento mais intenso de sua militância ocorreu durante a Ditadura Militar (1964-1985), que coibiu a organização política da sociedade civil. A Lei de Segurança Nacional, de setembro de 1967, estabelecia que era crime “incitar publicamente ao ódio ou à discriminação racial”. O que, na verdade, poderia ser usado contra o movimento negro, uma vez que denunciar o racismo e expor o mito da democracia racial poderia ser considerado uma ameaça à ordem social, um estímulo ao antagonismo e uma incitação ao preconceito. (Adaptado de Raquel Barreto, “Memória - Lélia Gonzalez”. Revista Cult 247. São Paulo, julho, 2019. Disponível em https://revistacult.uol.com.br/home/ lelia- gonzalez-perfil/. Acessado em 01/05/2020.) A partir do excerto sobre Lélia Gonzalez e seu contexto histórico, assinale a alternativa correta.",
        "subject": ["history"],
    },
    {
        "query": "Enunciado: O Programa Mundial de Alimentos da Organização das Nações Unidas (PMA-ONU) foi agraciado com o prêmio Nobel da Paz em 2020. No Brasil, um dos maiores produtores de alimentos do mundo, quatro em cada 10 famílias não tiveram acesso diário, regular, e permanente à quantidade suficiente de comida em 2017 e 2018. A fome é declarada quando a desnutrição é generalizada e quando as pessoas começam a morrer por falta de alimentos nutritivos e suficientes. A diversidade dos alimentos ingeridos garante nutrientes para o desempenho ideal das funções do organismo. (Fonte: UNITED NATIONS [UN]. World Food Program. What is famine? Disponível em https://www.wfp.org/stories/what-is-famine. Acessado em 08/06/ 2021.) Assinale a alternativa correta sobre os nutrientes e sua importância para a saúde humana.\nAlternativas:\nA. A hidrólise dos carboidratos essenciais fornece aminoácidos para a formação das proteínas, as quais têm função construtora de diferentes tecidos.\nB. Os lipídios contêm desoxirriboses e ácidos graxos, constituem as membranas plasmáticas e participam da síntese de colesterol no organismo.\nC. Os sais minerais são substâncias inorgânicas essenciais para diversas funções do organismo, como a síntese de glicogênio, de proteínas e de vitaminas.\nD. As vitaminas atuam como antioxidantes e são substâncias energéticas cuja composição fornece ao organismo glicídios utilizados na respiração celular.\nResposta:",
        "choices": [
            "A. A hidrólise dos carboidratos essenciais fornece aminoácidos para a formação das proteínas, as quais têm função construtora de diferentes tecidos.",
            "B. Os lipídios contêm desoxirriboses e ácidos graxos, constituem as membranas plasmáticas e participam da síntese de colesterol no organismo.",
            "C. Os sais minerais são substâncias inorgânicas essenciais para diversas funções do organismo, como a síntese de glicogênio, de proteínas e de vitaminas.",
            "D. As vitaminas atuam como antioxidantes e são substâncias energéticas cuja composição fornece ao organismo glicídios utilizados na respiração celular.",
        ],
        "gold": 2,
        "id": "UNICAMP_2022_65",
        "exam": "2022",
        "BK": False,
        "contents": "O Programa Mundial de Alimentos da Organização das Nações Unidas (PMA-ONU) foi agraciado com o prêmio Nobel da Paz em 2020. No Brasil, um dos maiores produtores de alimentos do mundo, quatro em cada 10 famílias não tiveram acesso diário, regular, e permanente à quantidade suficiente de comida em 2017 e 2018. A fome é declarada quando a desnutrição é generalizada e quando as pessoas começam a morrer por falta de alimentos nutritivos e suficientes. A diversidade dos alimentos ingeridos garante nutrientes para o desempenho ideal das funções do organismo. (Fonte: UNITED NATIONS [UN]. World Food Program. What is famine? Disponível em https://www.wfp.org/stories/what-is-famine. Acessado em 08/06/ 2021.) Assinale a alternativa correta sobre os nutrientes e sua importância para a saúde humana.",
        "subject": ["biology"],
    },
]


class BLUEX(MultipleChoicePromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "portuguese-benchmark-datasets/BLUEX"
    DATASET_NAME = None

    KEYS_TO_INDEX = ["context", "question"]
    SEARCHER_K = 10

    tag = None

    years_to_keep = None
    manual_examples = manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset = load_dataset(self.DATASET_PATH)["questions"]

        self.dataset = collections.defaultdict(list)

        for example in dataset:
            question_data = example.copy()
            year = example["id"].split("_")[1]

            if self.years_to_keep and int(year) not in self.years_to_keep:
                continue

            question_data["test_id"] = year

            # skip questions with null answers ( i.e the question was cancelled in the original test)
            if question_data["answer"] == None:
                continue

            # skip questions with associated images
            if question_data["has_associated_images"]:
                continue

            # substitute multiple spaces to single space

            for pattern, replace in PATTERNS_REPLACES:
                question_data["question"] = re.sub(
                    pattern, replace, question_data["question"]
                )
                for question_id in range(len(question_data["alternatives"])):
                    question_data["alternatives"][question_id] = re.sub(
                        pattern, replace, question_data["alternatives"][question_id]
                    )

            self.dataset["test"].append(question_data)

        self.dataset["test"] = list(map(self._process_doc, self.dataset["test"]))

    def create_collection_to_index(self):
        """Creates a JSON collection to index. Overwrite this funtion to keep
        more arguments.
        """
        with open(self.documents_to_index, "w") as f:
            json.dump(self.dataset["test"], f)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["test"]

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
            Enunciado: <enunciado>
            nAlternativas:
            A. <Alternativa1>
            B. <Alternativa2>
            C. <Alternativa3>
            D. <Alternativas4>
            Resposta:
            """
            prompt = "Enunciado: " + doc["question"] + "\nAlternativas:\n"
            for alternative in choices:
                prompt += f"{alternative}\n"
            prompt += "Resposta:"
            return prompt

        university = doc["id"].split("_")[0]
        if university == "UNICAMP":
            choices = ["a", "b", "c", "d"]
        else:
            choices = ["a", "b", "c", "d", "e"]

        alternative_mapping = {
            "a)": "A.",
            "b)": "B.",
            "c)": "C.",
            "d)": "D.",
            "e)": "E.",

        }

        alternatives = []
        for alternative, mapping in zip(
            doc["alternatives"], alternative_mapping.items()
        ):
            key, value = mapping
            alternatives.append(alternative.replace(key, value))
        return {
            "query": format_example(doc, alternatives),
            "choices": alternatives,
            "gold": choices.index(doc["answer"].lower()),
            "id": doc["id"],
            "exam": doc["test_id"],
            # BK stands for Brazilian Knowledge and is true when the question
            # is strongly related to the Brazilian history, culture, geography, literature, etc.
            "BK": doc["BK"],
            "contents": doc["question"],  # used for indexing
            "subject": doc["subject"] if "subject" in doc else [],
            "university": university,
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        results = {
            "acc": acc,
            "acc_norm": acc_norm,
            doc["exam"]: acc_norm,
        }

        if doc["BK"]:
            results["BK"] = acc_norm

        if doc["university"]:
            results[doc["university"]] = acc

        for sub in doc["subject"]:
            results[sub] = acc_norm

        return results

    def higher_is_better(self):
        years = ["2024", "2023", "2022", "2021", "2020", "2019", "2018"]
        subjects = [
            "portuguese",
            "mathematics",
            "history",
            "physics",
            "chemistry",
            "geography",
            "biology",
            "english",
            "philosophy",
        ]

        years_agg_dict = {year: True for year in years}
        subjects_agg_dict = {subject: True for subject in subjects}

        return {
            "acc": True,
            "acc_norm": True,
            "BK": True,
            "USP": True,
            "UNICAMP": True,
            **years_agg_dict,
            **subjects_agg_dict,
        }

    def aggregation(self):
        years = ["2024", "2023", "2022", "2021", "2020", "2019", "2018"]
        subjects = [
            "portuguese",
            "mathematics",
            "history",
            "physics",
            "chemistry",
            "geography",
            "biology",
            "english",
            "philosophy",
        ]

        years_agg_dict = {year: mean for year in years}
        subjects_agg_dict = {subject: mean for subject in subjects}

        return {
            "acc": mean,
            "acc_norm": mean,
            "BK": mean,
            "USP": mean,
            "UNICAMP": mean,
            "unknown_pred": mean,
            **years_agg_dict,
            **subjects_agg_dict,
        }

    def fewshot_examples(self, k, rnd, prompt_mode, doc):
        # For each doc, limit the self._training_docs to examples from other exams.
        # We also remove the top-10 largest documents from the list of prompt candidates.
        self._training_docs = []
        for d in self.training_docs():
            if d["exam"] != doc["exam"]:
                self._training_docs.append(d)
        if prompt_mode == "dynamic-random":
            return rnd.sample(self._training_docs, k)

        elif prompt_mode == "fixed":
            return rnd.sample(self._training_docs[:k], k)

        elif prompt_mode == "manual":
            _manual_docs = []
            for d in self.manual_examples:
                if d["exam"] != doc["exam"]:
                    _manual_docs.append(d)
            assert k <= len(_manual_docs), (
                f"The number of manual_examples is not enough to satisfy "
                f"num_fewshot={k}. Please, include more examples."
            )
            return rnd.sample(_manual_docs, k)

        else:
            print(
                'Please set prompt_mode as "fixed", "dynamic-random"'
            )


class BLUEX_GREEDY(BLUEX):
    def doc_to_target(self, doc):
        return " " + ["A.", "B.", "C.", "D.", "E."][doc["gold"]]

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
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document
        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = ["A.", "B.", "C.", "D.", "E."][doc["gold"]].strip()
        pred = results[0].strip()
        unknown=False
        # regex processing.
        match_1 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\.", pred)
        match_2 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDEabcde])\)", pred)
        match_3 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])", pred)
        if len(match_1) > 0:
            pred = match_1[-1] + "."
        elif len(match_2) > 0:
            pred = match_2[-1].upper() + "."
        elif len(match_3) > 0:
            pred = match_3[-1] + "."
        # if the pred matches an alternative text, convert to respective letter
        elif pred in doc['choices']:
            ind = doc['choices'].index(pred)
            pred = ["A.", "B.", "C.", "D.", "E."][ind]
        else:
            print(f"Regex failed at processing {pred}")
            pred=""
            unknown = True

        acc = 1.0 if pred == gold else 0.0

        debug_info = {
            "gold": gold,
            "pred": pred,
        }

        results = {
            "acc": acc,
            doc["exam"]: acc,
            "debug_info": debug_info,
            "unknown_pred": 1.0 if unknown else 0.0,
        }

        if doc["BK"]:
            results["BK"] = acc

        if doc["university"]:
            results[doc["university"]] = acc

        for sub in doc["subject"]:
            results[sub] = acc

        return results


class BLUEX_RECENT(BLUEX):
    years_to_keep = [2023, 2024]

class BLUEX_LAUNCH_VERSION_GREEDY(BLUEX_GREEDY):
    # bluex used in v1 had tests up to 2023
    years_to_keep = [2018,2019,2020,2021,2022,2023]

class BLUEX_RECENT_GREEDY(BLUEX_GREEDY):
    years_to_keep = [2023, 2024]

class BLUEX_2024_GREEDY(BLUEX_GREEDY):
    years_to_keep = [2024]

class BLUEX_UNICAMP_2024_GREEDY(BLUEX_2024_GREEDY):
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        # Filtrar apenas para a universidade "UNICAMP"
        self.dataset["test"] = [
            example for example in self.dataset["test"]
            if example["university"] == "UNICAMP"
        ]

class BLUEX_USP_2024_GREEDY(BLUEX_2024_GREEDY):
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        # Filtrar apenas para a universidade "USP"
        self.dataset["test"] = [
            example for example in self.dataset["test"]
            if example["university"] == "USP"
        ]
