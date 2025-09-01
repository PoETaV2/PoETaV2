"""
POSCOMP is a dataset containing multiple choice questions extracted from pos graduate exams for people majored in computer related fields in brazil.
"""

import collections
import json
import numpy as np
import os
import re
from datasets import load_dataset

from lm_eval.base import rf, MultipleChoicePromptSelectionTask
from lm_eval.metrics import mean


_CITATION = """
Em breve?
"""


PATTERNS_REPLACES = [
    (r"^\s+", r""),
]


apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)


manual_examples = [
    {
        "query": "Enunciado: Determine o valor de x para que o vetor (1, x, 5) ∈ R³ pertença ao subespaço <(1, 2, 3), (1, 1, 1)>\nAlternativas:\nA. x = 0\nB. x = -1\nC. x = 1\nD. x = 3\nE. x = 7\nResposta:",
        "choices": ["A. x = 0", "B. x = -1", "C. x = 1", "D. x = 3", "E. x = 7"],
        "gold": 3,
        "id": "POSCOMP_2023_2",
        "exam": "2023",
        "MR": True,
        "CR": False,
        "contents": "Determine o valor de x para que o vetor (1, x, 5) ∈ R³ pertença ao subespaço <(1, 2, 3), (1, 1, 1)>",
        "subject": ["mathematics"],
    },
    {
        "query": "Enunciado: Qual é a implementação no qual um grafo G = (V,A) contendo n vértices é uma matriz n x n de bits, em que A[i,j] é 1 (ou verdadeiro, no caso de booleanos) se e somente se existe um arco do vértice i para o vértice j.\nAlternativas:\nA. Matriz de incidência.\nB. Lista de adjacência.\nC. Matriz de adjacência.\nD. Lista de incidência.\nE. Matriz quadrada completa.\nResposta:",
        "choices": [
            "A. Matriz de incidência.",
            "B. Lista de adjacência.",
            "C. Matriz de adjacência.",
            "D. Lista de incidência.",
            "E. Matriz quadrada completa.",
        ],
        "gold": 2,
        "id": "POSCOMP_2022_36",
        "exam": "2022",
        "MR": False,
        "CR": True,
        "contents": "Qual é a implementação no qual um grafo G = (V,A) contendo n vértices é uma matriz n x n de bits, em que A[i,j] é 1 (ou verdadeiro, no caso de booleanos) se e somente se existe um arco do vértice i para o vértice j.",
        "subject": ["graph_theory", "computer_fundamentals"],
    },
    {
        "query": "Enunciado: Resolva o sistema de equações lineares pelo método de Gauss, se a matriz do sistema é: \n\n1 2 −3 | −2 \n3 0 1 | 0 \n2 −1 2 | 3\n\nAlternativas:\nA. x= 1; y = -9; z = 6\nB. x = 2; y = -11; z = -6\nC. x= 1; y = 2; z = -3\nD. x= -2; y = 6; z = 3\nE. x= -2; y = 6; z = -6\nResposta:",
        "choices": [
            "A. x= 1; y = -9; z = 6",
            "B. x = 2; y = -11; z = -6",
            "C. x= 1; y = 2; z = -3",
            "D. x= -2; y = 6; z = 3",
            "E. x= -2; y = 6; z = -6",
        ],
        "gold": 1,
        "id": "POSCOMP_2022_7",
        "exam": "2022",
        "MR": True,
        "CR": False,
        "contents": "Resolva o sistema de equações lineares pelo método de Gauss, se a matriz do sistema é: \n\n1 2 −3 | −2 \n3 0 1 | 0 \n2 −1 2 | 3\n\n",
        "subject": ["linear_algebra", "mathematics"],
    },
    {
        "query": "Enunciado: Em relação aos métodos de interpolação de intensidade de níveis de cinza ou cor de uma imagem, analise as assertivas abaixo e assinale V, se verdadeiras, ou F, se falsas. \n( ) O método do vizinho mais próximo atribui a cada nova posição a intensidade de seu vizinho mais próximo na imagem original. O método pode causar distorções em detalhes finos ou criar formas serrilhadas em bordas retas de imagens. \n( ) Na interpolação bilinear, os dois vizinhos mais próximos são utilizados para estimar a intensidade de uma dada posição. O método se baseia na média aritmética de distância desses pixels e causa borramento devido à sua característica de suavização. \n( ) A interpolação bicúbica inclui os dezesseis vizinhos mais próximos de um ponto. Esse tipo de interpolação preserva detalhes finos na imagem. \nA ordem correta de preenchimento dos parênteses, de cima para baixo, é:\nAlternativas:\nA. F – F – V.\nB. F – V – F.\nC. V – F – V.\nD. V – V – V.\nE. V – V – F.\nResposta:",
        "choices": [
            "A. F – F – V.",
            "B. F – V – F.",
            "C. V – F – V.",
            "D. V – V – V.",
            "E. V – V – F.",
        ],
        "gold": 2,
        "id": "POSCOMP_2023_63",
        "exam": "2023",
        "MR": False,
        "CR": False,
        "contents": "Em relação aos métodos de interpolação de intensidade de níveis de cinza ou cor de uma imagem, analise as assertivas abaixo e assinale V, se verdadeiras, ou F, se falsas. \n( ) O método do vizinho mais próximo atribui a cada nova posição a intensidade de seu vizinho mais próximo na imagem original. O método pode causar distorções em detalhes finos ou criar formas serrilhadas em bordas retas de imagens. \n( ) Na interpolação bilinear, os dois vizinhos mais próximos são utilizados para estimar a intensidade de uma dada posição. O método se baseia na média aritmética de distância desses pixels e causa borramento devido à sua característica de suavização. \n( ) A interpolação bicúbica inclui os dezesseis vizinhos mais próximos de um ponto. Esse tipo de interpolação preserva detalhes finos na imagem. \nA ordem correta de preenchimento dos parênteses, de cima para baixo, é:",
        "subject": ["computer_technology"],
    },
    {
        "query": "Enunciado: Requisitos não funcionais envolvem requisitos de produto, organizacionais e externos (SOMMERVILLE, 2011). Os requisitos de produto especificam ou restringem o funcionamento do software. Os organizacionais atendem a políticas ou procedimentos relativos aos clientes e/ou organizações. Já os requisitos externos são derivados de fatores externos ao sistema e ao processo de desenvolvimento. Considere as subclasses de requisitos não funcionais abaixo, e os respectivos exemplos. \n\n- Requisitos de Ambiente, tal como a necessidade de o sistema funcionar em determinados sistemas operacionais. \n- Requisitos de Legislação, tal como o direito dos pacientes à privacidade em um sistema médico. \n- Requisitos de Usabilidade, tal como acessibilidade por pessoas com deficiências. \nClassifique estas subclasses de acordo com os três tipos de requisitos não funcionais, considerando a ordem de cima para baixo.\nAlternativas:\nA. Produto – Organizacional – Externo.\nB. Organizacional – Externo – Externo.\nC. Produto – Organizacional – Produto.\nD. Organizacional – Externo – Produto.\nE. Produto – Externo – Produto.\nResposta:",
        "choices": [
            "A. Produto – Organizacional – Externo.",
            "B. Organizacional – Externo – Externo.",
            "C. Produto – Organizacional – Produto.",
            "D. Organizacional – Externo – Produto.",
            "E. Produto – Externo – Produto.",
        ],
        "gold": 3,
        "id": "POSCOMP_2022_56",
        "exam": "2022",
        "MR": False,
        "CR": False,
        "contents": "Requisitos não funcionais envolvem requisitos de produto, organizacionais e externos (SOMMERVILLE, 2011). Os requisitos de produto especificam ou restringem o funcionamento do software. Os organizacionais atendem a políticas ou procedimentos relativos aos clientes e/ou organizações. Já os requisitos externos são derivados de fatores externos ao sistema e ao processo de desenvolvimento. Considere as subclasses de requisitos não funcionais abaixo, e os respectivos exemplos. \n\n- Requisitos de Ambiente, tal como a necessidade de o sistema funcionar em determinados sistemas operacionais. \n- Requisitos de Legislação, tal como o direito dos pacientes à privacidade em um sistema médico. \n- Requisitos de Usabilidade, tal como acessibilidade por pessoas com deficiências. \nClassifique estas subclasses de acordo com os três tipos de requisitos não funcionais, considerando a ordem de cima para baixo.",
        "subject": ["software_engineering", "computer_technology"],
    },
    {
        "query": "Enunciado: Sobre funções Hash, é correto afirmar que:\nAlternativas:\nA. O método de divisão funciona em duas etapas. Na primeira etapa, multiplica-se a chave k por \numa constante A na faixa 0<A<1 e extrai-se a parte fracionária de kA. Na segunda etapa, \nmultiplica-se esse valor por m e toma-se o piso do resultado.\nB. Em endereçamento aberto, todos os elementos ficam na própria tabela de espelhamento. Isto é, \ncada entrada da tabela contém um elemento do conjunto dinâmico ou NIL. Ao procurar um \nelemento, examina-se sistematicamente as posições da tabela até encontrar o elemento desejado \nou até confirmar que o elemento não está na tabela.\nC. No método de encadeamento não existe nenhuma lista e nenhum elemento fora da tabela.\nD. O hashing pode proporcionar excelente desempenho no pior caso, quando o conjunto de chaves \né dinâmico, isto é, assim que as chaves são armazenadas na tabela, o conjunto de chaves muda \nautomaticamente de tempos em tempos.\nE. No método de multiplicação, mapeia-se uma chave k para uma de m posições, tomando o resto \nda divisão de k por m.\nResposta:",
        "choices": [
            "A. O método de divisão funciona em duas etapas. Na primeira etapa, multiplica-se a chave k por \numa constante A na faixa 0<A<1 e extrai-se a parte fracionária de kA. Na segunda etapa, \nmultiplica-se esse valor por m e toma-se o piso do resultado.",
            "B. Em endereçamento aberto, todos os elementos ficam na própria tabela de espelhamento. Isto é, \ncada entrada da tabela contém um elemento do conjunto dinâmico ou NIL. Ao procurar um \nelemento, examina-se sistematicamente as posições da tabela até encontrar o elemento desejado \nou até confirmar que o elemento não está na tabela.",
            "C. No método de encadeamento não existe nenhuma lista e nenhum elemento fora da tabela.",
            "D. O hashing pode proporcionar excelente desempenho no pior caso, quando o conjunto de chaves \né dinâmico, isto é, assim que as chaves são armazenadas na tabela, o conjunto de chaves muda \nautomaticamente de tempos em tempos.",
            "E. No método de multiplicação, mapeia-se uma chave k para uma de m posições, tomando o resto \nda divisão de k por m.",
        ],
        "gold": 1,
        "id": "POSCOMP_2023_24",
        "exam": "2023",
        "MR": False,
        "CR": True,
        "contents": "Sobre funções Hash, é correto afirmar que:",
        "subject": ["computer_fundamentals"],
    },
]


class POSCOMP(MultipleChoicePromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "maritaca-ai/poscomp"
    DATASET_NAME = None

    KEYS_TO_INDEX = ["context", "question"]
    SEARCHER_K = 10

    tag = None

    years_to_keep = None
    manual_examples = manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset = load_dataset(
            self.DATASET_PATH
        )["train"]

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

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
            Enunciado: <enunciado>
            Alternativas:
            A. <Alternativa1>
            B. <Alternativa2>
            C. <Alternativa3>
            D. <Alternativa4>
            E. <Alternativa4>
            Resposta:
            """
            prompt = "Enunciado: " + doc["question"] + "\nAlternativas:\n"
            for alternative in choices:
                prompt += f"{alternative}\n"
            prompt += "Resposta:"
            return prompt

        choices = ["a", "b", "c", "d", "e"]

        alternative_mapping = {
            "A)": "A.",
            "B)": "B.",
            "C)": "C.",
            "D)": "D.",
            "E)": "E.",
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
            "MR": doc["MR"],
            "CR": doc["CR"],
            "contents": doc["question"],  # used for indexing
            "subject": doc["subject"] if "subject" in doc else [],
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

        for sub in doc["subject"]:
            if sub in ["computer_technology", "mathematics", "computer_fundamentals"]:
                results[sub] = acc_norm

        if doc["MR"]:
            results["MR"] = acc_norm

        if doc["CR"]:
            results["CR"] = acc_norm

        return results

    def higher_is_better(self):
        years = ["2023", "2022"]
        subjects = ["computer_technology", "mathematics", "computer_fundamentals"]

        years_agg_dict = {year: True for year in years}
        subjects_agg_dict = {subject: True for subject in subjects}

        return {
            "acc": True,
            "acc_norm": True,
            "MR": True,
            "CR": True,
            **years_agg_dict,
            **subjects_agg_dict,
        }

    def aggregation(self):
        years = ["2023", "2022"]
        subjects = ["computer_technology", "mathematics", "computer_fundamentals"]

        years_agg_dict = {year: mean for year in years}
        subjects_agg_dict = {subject: mean for subject in subjects}

        return {
            "acc": mean,
            "acc_norm": mean,
            "MR": mean,
            "CR": mean,
            "unknown_pred": mean,
            **years_agg_dict,
            **subjects_agg_dict,
        }

    def fewshot_examples(self, k, rnd, prompt_mode, doc):
        # For each doc, limit the self._training_docs to examples from other exams.
        # We also remove the top-10 largest documents from the list of prompt candidates.
        self._training_docs = []
        for d in self.test_docs():
            if d["exam"] != doc["exam"]:
                self._training_docs.append(d)
        if prompt_mode == "dynamic-random":
            return rnd.sample(self._training_docs, k)

        elif prompt_mode == "fixed":
            return rnd.sample(self._test_docs[:k], k)

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

        elif prompt_mode == "dynamic-similar":
            if self.searcher is None:
                from pyserini.search.lucene import LuceneSearcher

                self.indexes_dir = os.path.join("data", self.DATASET_PATH, "indexes")
                self.documents_to_index = os.path.join(
                    "data", self.DATASET_PATH, "docs_to_index", "documents.json"
                )

                # Index the training documents for dynamic-similar prompt mode.
                # Run only once.
                if not os.path.exists(self.indexes_dir):
                    os.makedirs(os.path.dirname(self.documents_to_index), exist_ok=True)
                    self.create_collection_to_index()
                    self.indexing()

                # Instantiate the searcher for dynamic-similar prompts.
                self.searcher = LuceneSearcher(self.indexes_dir)
                self.searcher.set_language("pt")

            hits = self.searcher.search(doc["contents"], k=self.SEARCHER_K)
            selected_hits = []

            for hit in hits:
                hit = json.loads(hit.raw)

                if hit["exam"] != doc["exam"]:
                    selected_hits.append(hit)

                if len(selected_hits) == k:
                    break

            # check if we have enough similar examples. If not, complete with
            # random examples.
            i = 0
            while len(selected_hits) < k:
                if self._training_docs[i] not in selected_hits:
                    selected_hits.append(self._training_docs[i])
                i += 1

            # move the most relevant examples to the end.
            selected_hits.reverse()

            return selected_hits

        else:
            print(
                'Please set prompt_mode as "fixed", "dynamic-random", or "dynamic-similar"'
            )


class POSCOMP_GREEDY(POSCOMP):
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
        unknown = False
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
        elif pred in doc["choices"]:
            ind = doc["choices"].index(pred)
            pred = ["A.", "B.", "C.", "D.", "E."][ind]
        else:
            print(f"Regex failed at processing {pred}")
            unknown = True
            pred = ""

        acc = 1.0 if pred == gold else 0.0

        results = {
            "acc": acc,
            doc["exam"]: acc,
            "unknown_pred": 1.0 if unknown else 0.0,
            "debug_info": {
                "gold": gold,
                "pred": pred,
            },
        }

        for sub in doc["subject"]:
            if sub in ["computer_technology", "mathematics", "computer_fundamentals"]:
                results[sub] = acc

        if doc["MR"]:
            results["MR"] = acc

        if doc["CR"]:
            results["CR"] = acc

        return results


class POSCOMP_RECENT(POSCOMP):
    years_to_keep = [2023]


class POSCOMP_RECENT_GREEDY(POSCOMP_GREEDY):
    years_to_keep = [2023]
