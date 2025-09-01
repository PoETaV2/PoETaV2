"""
FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education
https://ieeexplore.ieee.org/document/8923668

The FaQuAD is a Portuguese reading comprehension dataset which follows the format 
of the Stanford Question Answering Dataset (SQuAD). As far as we know, FaQuAD is 
a pioneer Portuguese reading comprehension dataset with the SQuAD's challenging format.
"""
import os
import json
import numpy as np
import datasets
from urllib.request import urlretrieve
from conversation import get_conv_template
from math import exp
from lm_eval import utils
from lm_eval.base import rf, PromptSelectionTask
from functools import partial
from lm_eval.metrics import squad_exact_match, squad_f1


_CITATION = """
@INPROCEEDINGS{8923668,
  author={Sayama, Hélio Fonseca and Araujo, Anderson Viçoso and Fernandes, Eraldo Rezende},
  booktitle={2019 8th Brazilian Conference on Intelligent Systems (BRACIS)}, 
  title={FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education}, 
  year={2019},
  volume={},
  number={},
  pages={443-448},
  doi={10.1109/BRACIS.2019.00084}}
"""

_manual_examples=[
    {"title": "ATIVIDADES_COMPLEMENTARES", "context": "Realizar um curso de graduação é uma experiência bastante enriquecedora. Na verdade, tanto profissional, quanto pessoalmente, toda oportunidade de estudo proporciona conhecimento e experiência para avançar na carreira, além de trabalhar responsabilidade, comprometimento e habilidades sociais de uma pessoa. Os cursos de graduação compõe o ensino superior, e fazem partem de uma modalidade que, segundo a lei, tem que ser fiscalizada e regulamentada pelo MEC (Ministério da Educação e Cultura). Mas por serem fiscalizados, quer dizer que os cursos de graduação são os melhores entre todos os diferentes tipos de cursos? Definitivamente não.", "question": "Quem fiscaliza o ensino superior?", "id": "5d8dcc81623941dea803ac4eec578623", "answers": {"answer_start": [455, 455, 460], "text": ["MEC", "MEC (Ministério da Educação e Cultura)", "Ministério da Educação e Cultura"]}},
    {"title": "POS_GRADUACAO", "context": "Os cursos de pós-graduação lato sensu, em nível de especialização, são regulados pela Resolução CNE/CES nº 1, de 8 de junho de 2007. A duração mínima desses cursos é de 360 horas, além do tempo destinado à elaboração de monografia ou trabalho de conclusão de curso. A especialização dá oportunidade, ao graduado, de prosseguir seus estudos ao se habilitar à docência e se especializar em áreas do conhecimento voltadas ao mundo do trabalho, podendo ser uma área diretamente ligada à primeira graduação ou não. Em alguns países, os créditos dos certificados lato sensu podem contar como o primeiro ano de um mestrado na mesma área.", "question": "Qual nível dos cursos de pós-graduação são regulados pela resolução CNE/CES nº 1?", "id": "7994bd1e979c4d3dab089cb99a79aa51", "answers": {"answer_start": [51, 51, 51], "text": ["especialização", "especialização", "especialização"]}},
    {"title": "COMPUTADOR", "context": "Hoje em dia, muitos computadores aparentam executar vários programas ao mesmo tempo, o que é normalmente conhecido como multitarefa. Na realidade, a CPU executa as instruções de um programa por um curto período de tempo e, em seguida, troca para um outro programa e executa algumas de suas instruções. Isto cria a ilusão de vários programas sendo executados simultaneamente através do compartilhamento do tempo da CPU entre os programas. Este compartilhamento de tempo é normalmente controlado pelo sistema operacional. Nos casos em que o computador possui dois núcleos de processamento, cada núcleo processa informações de um programa, diminuindo assim o tempo de processamento.", "question": "Como a CPU executa as instruções de um programa?", "id": "77c2e4b820704cd9ad5d07ef366a5994", "answers": {"answer_start": [190, 147, 153], "text": ["por um curto período de tempo e, em seguida, troca para um outro programa e executa algumas de suas instruções", "a CPU executa as instruções de um programa por um curto período de tempo e, em seguida, troca para um outro programa e executa algumas de suas instruções", "executa as instruções de um programa por um curto período de tempo e, em seguida, troca para um outro programa e executa algumas de suas instruções"]}},
    {"title": "ENGENHARIA_DE_COMPUTACAO", "context": "O curso de graduação em Engenharia de Computação tem sido adicionado às universidades desde o início dos anos 1990. Algumas universidades como o Instituto de Tecnologia de Massachusetts (MIT), nos Estados Unidos, optaram por unir os departamentos de engenharia elétrica e de ciência da computação. ", "question": "Onde fica o MIT?", "id": "527cc7b20ff94b538841a3fc0e52822c", "answers": {"answer_start": [197, 197, 197], "text": ["Estados Unidos", "Estados Unidos", "Estados Unidos"]}},
]

# def _squad_metric(predictions, references):
#     squad_metric = datasets.load_metric("lm_eval/custom_eval/squad/squad.py")
#     return squad_metric.compute(predictions=predictions, references=references)


# def _squad_agg(key, items):
#     predictions, references = zip(*items)

#     return _squad_metric(predictions=predictions, references=references)[key]


class FAQuAD(PromptSelectionTask):
    VERSION = 1
    DATASET_PATH = "data/faquad/"

    KEYS_TO_INDEX = ['question']
    SEARCHER_K = 10
    
    manual_examples = _manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        if os.path.exists(self.DATASET_PATH):
            print(f"Reusing dataset faquad ({self.DATASET_PATH})")
        else:
            os.makedirs(self.DATASET_PATH, exist_ok=True)
            urlretrieve('https://raw.githubusercontent.com/liafacom/faquad/master/data/train.json', os.path.join(self.DATASET_PATH, "train.json"))
            urlretrieve('https://raw.githubusercontent.com/liafacom/faquad/master/data/dev.json', os.path.join(self.DATASET_PATH, "dev.json"))

        self.dataset = {}
        self.dataset['train'] = [ sample for id, sample in self._generate_examples(self.DATASET_PATH + 'train.json')]
        self.dataset['validation'] = [ sample for id, sample in self._generate_examples(self.DATASET_PATH + 'dev.json')]

        # To maximize the number of few-shot examples we can use in the context, 
        # we filtered out nearly half of the training examples. We removed those
        # whose number of tokens (self.doc_to_text(doc) + ' ' + doc['answers']['text'][0])
        # surpasses 357. We did not use the median (355) in order to keep first 8 examples). 
        self.train_indices_to_keep_as_candidates = [0, 1, 2, 3, 4, 5, 6, 7, 22, 
            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 
            65, 66, 67, 68, 69, 76, 77, 78, 79, 80, 81, 82, 83, 84, 90, 91, 92, 
            93, 94, 95, 96, 97, 102, 103, 104, 105, 106, 107, 108, 109, 113, 
            115, 116, 117, 118, 119, 120, 121, 126, 127, 128, 129, 135, 136, 
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 
            150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 
            167, 168, 169, 170, 185, 186, 205, 206, 207, 208, 209, 210, 211, 
            212, 213, 215, 216, 217, 218, 220, 239, 240, 241, 242, 243, 244, 
            245, 246, 247, 248, 249, 250, 255, 256, 257, 258, 259, 260, 261, 
            262, 263, 264, 265, 266, 270, 272, 274, 275, 276, 277, 278, 279, 
            298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 
            311, 322, 323, 324, 325, 326, 327, 328, 329, 331, 335, 336, 337, 
            338, 339, 340, 341, 342, 343, 344, 345, 346, 351, 352, 353, 354, 
            355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 372, 373, 374, 
            375, 376, 377, 381, 382, 383, 384, 388, 389, 390, 391, 428, 429, 
            430, 431, 432, 433, 434, 448, 449, 450, 451, 461, 462, 463, 464, 
            465, 469, 491, 492, 494, 495, 496, 497, 498, 499, 500, 504, 505, 
            506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 
            519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 
            532, 543, 546, 549, 551, 552, 553, 554, 555, 556, 557, 558, 559, 
            560, 561, 562, 563, 564, 565, 566, 567, 568, 570, 571, 572, 573, 
            574, 582, 583, 584, 585, 589, 591, 592, 593, 594, 595, 619, 620, 
            621, 622, 623, 624, 625, 626, 627, 639, 640, 641, 642, 643, 644, 
            645, 646, 647, 648, 649, 650, 652, 654, 657, 658, 659, 660, 661, 
            662, 663, 664, 684, 685, 686, 687, 688, 689, 690, 702, 704, 706, 
            716, 717, 718, 719, 720, 735, 736, 742, 743, 744, 745, 746, 747, 
            748, 749, 757, 758, 759, 763, 765, 766, 767, 768, 769, 770, 771, 
            772, 776, 777, 778, 779, 780, 781, 782, 783, 784, 794, 795, 796, 
            797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 
            810, 811, 812, 813, 814, 815, 816, 821, 822, 823, 834, 835, 836]
        
        self.dataset['train'] = np.array(self.dataset['train'])[self.train_indices_to_keep_as_candidates]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        print(f"Generating examples from {filepath}")
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return 'Contexto: ' + doc['context'] + '\n' + 'Pergunta: ' + doc['question'] + '\n' + 'Resposta:'

    def doc_to_target(self, doc):
        return " " + doc['answers']['text'][0]

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        continuation = rf.greedy_until(ctx, ['\n'])
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
        predictions = results[0]

        references = doc['answers']['text']

        return { 
            'exact': (predictions, references), # Exact match (the normalized answer exactly match the gold answer)
            'f1': (predictions, references), #  The F-score of predicted tokens versus the gold answer
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return { 
            'exact': squad_exact_match, # Exact match (the normalized answer exactly match the gold answer)
            'f1': squad_f1, #  The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return { 
            'exact': True, # Exact match (the normalized answer exactly match the gold answer)
            'f1': True, #  The F-score of predicted tokens versus the gold answer
        }

    @utils.positional_deprecated
    def fewshot_context(
        self, doc, num_fewshot, prompt_mode=None, provide_description=None, rnd=None, description=None, conversation_template=None, **kwargs
    ):
        # Those documents and respective dynamic contexts do not fit into 
        # 2048-150 tokens. Decreasing the num_fewshot.
        if prompt_mode != 'fixed' and doc['id'] in \
            ['bd6b3b2b1c7b4c49a3affa5ac5b54f31', '568a6fa71815493e865cc15dc4ea5a26'] and num_fewshot>0:
            num_fewshot -= 1

        return super().fewshot_context(
           doc=doc, num_fewshot=num_fewshot, prompt_mode=prompt_mode,
           provide_description=provide_description, rnd=rnd, 
           description=description, conversation_template=conversation_template, **kwargs
        )
    
