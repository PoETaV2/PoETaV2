"""
The ASSIN 2 Shared Task: A Quick Overview
https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39

The ASSIN 2 corpus is composed of rather simple sentences. Following the 
procedures of SemEval 2014 Task 1. The training and validation data are composed, 
respectively, of 6,500 and 500 sentence pairs in Brazilian Portuguese, annotated 
for entailment and semantic similarity. Semantic similarity values range from 1 
to 5, and text entailment classes are either entailment or none. The test data 
are composed of approximately 3,000 sentence pairs with the same annotation. 
All data were manually annotated.

Homepage: "https://sites.google.com/view/assin2/"
"""
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from lm_eval.base import rf, PromptSelectionTask
from ..metrics import mean


_CITATION = """
@inproceedings{real2020assin,
    title = "The assin 2 shared task: a quick overview",
    author= "Real, Livy and
        Fonseca, Erick and 
        Oliveira, Hugo Goncalo",
    booktitle = "International Conference on Computational Processing of the Portuguese Language",
    pages = "406--412",
    year = "2020,
    organization = "Springer",
}
"""

_manual_examples=[
    {"sentence_pair_id": 120, "premise": "Um casal está sentado em um pátio e está encarando o oceano", "hypothesis": "Um casal está sentado em um pátio e olhando para o oceano", "relatedness_score": 4.699999809265137, "entailment_judgment": 1},
    {"sentence_pair_id": 141, "premise": "Um homem está lendo o e-mail", "hypothesis": "Alguém está verificando o e-mail", "relatedness_score": 4.065000057220459, "entailment_judgment": 1},
    {"sentence_pair_id": 149, "premise": "A mulher está fatiando alho", "hypothesis": "A mulher está picando alho", "relatedness_score": 4.699999809265137, "entailment_judgment": 1},
    {"sentence_pair_id": 160, "premise": "O cachorro está sendo levado para passear pela mulher", "hypothesis": "A mulher está passeando com o cachorro", "relatedness_score": 4.800000190734863, "entailment_judgment": 1},
    {"sentence_pair_id": 211, "premise": "Um gambá está sendo segurado por uma pessoa", "hypothesis": "Uma pessoa está segurando um gambá", "relatedness_score": 5.0, "entailment_judgment": 1},
    {"sentence_pair_id": 235, "premise": "Um chefe está limpando uma tigela de sopa", "hypothesis": "Uma tigela de sopa está sendo limpa por um chefe", "relatedness_score": 4.800000190734863, "entailment_judgment": 1},
    {"sentence_pair_id": 703, "premise": "A senhora está medindo o tornozelo de outra mulher", "hypothesis": "Uma mulher está sendo medida por outra mulher", "relatedness_score": 4.400000095367432, "entailment_judgment": 1},
    {"sentence_pair_id": 861, "premise": "Um homem está telefonando", "hypothesis": "Um homem está conversando ao telefone", "relatedness_score": 4.300000190734863, "entailment_judgment": 1},
    {"sentence_pair_id": 891, "premise": "Um cachorro está pegando um pedaço de madeira da água clara", "hypothesis": "Um cachorro está pegando uma vara de dentro de uma água muito limpa", "relatedness_score": 4.699999809265137, "entailment_judgment": 1},
    {"sentence_pair_id": 4583, "premise": "Algumas crianças estão pulando em um trampolim", "hypothesis": "Ninguém está brincando em um trampolim", "relatedness_score": 3.799999952316284, "entailment_judgment": 0},
    {"sentence_pair_id": 4605, "premise": "Um homem está entrando em um carro em uma garagem", "hypothesis": "Um homem está estacionando um carro em uma garagem", "relatedness_score": 3.5999999046325684, "entailment_judgment": 0},
    {"sentence_pair_id": 4666, "premise": "Um homem está tocando um trompete", "hypothesis": "O violão está sendo tocado pelo homem", "relatedness_score": 3.0, "entailment_judgment": 0},
    {"sentence_pair_id": 4687, "premise": "Um homem está andando a cavalo", "hypothesis": "O homem não está montando um cavalo", "relatedness_score": 4.300000190734863, "entailment_judgment": 0},
    {"sentence_pair_id": 4803, "premise": "Um homem está caminhando na floresta", "hypothesis": "O homem está andando alegremente na floresta", "relatedness_score": 4.199999809265137, "entailment_judgment": 0},
    {"sentence_pair_id": 4832, "premise": "A pessoa está tocando a flauta", "hypothesis": "O homem está tocando a flauta", "relatedness_score": 4.699999809265137, "entailment_judgment": 0},
    {"sentence_pair_id": 4880, "premise": "Crianças de camisas vermelhas estão dormindo nas folhas", "hypothesis": "Crianças em camisas vermelhas estão brincando nas folhas", "relatedness_score": 4.0, "entailment_judgment": 0},
    {"sentence_pair_id": 4914, "premise": "As pessoas estão andando de bote e remando", "hypothesis": "Ninguém está flutuando em um bote", "relatedness_score": 4.099999904632568, "entailment_judgment": 0},
    {"sentence_pair_id": 5063, "premise": "Os gatos estão brincando um com o outro", "hypothesis": "Dois gatos estão deitados juntos", "relatedness_score": 3.299999952316284, "entailment_judgment": 0},
]

class ASSIN_RTE(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "assin2"
    DATASET_NAME = None

    KEYS_TO_INDEX = ['premise', 'hypothesis']
    KEY_TO_BALANCE = 'entailment_judgment'
    NUM_CLASSES = 2
    SEARCHER_K = 600

    manual_examples=_manual_examples

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Pergunta: Dado que \"{doc['premise']}\", é verdade que \"{doc['hypothesis']}\"?\nResposta:"

    def doc_to_target(self, doc):
        return " " + ["Não", "Sim"][doc['entailment_judgment']]

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
        ll_none, _ = rf.loglikelihood(ctx, " Não") 
        ll_entailment, _ = rf.loglikelihood(ctx, " Sim") 
        return ll_none, ll_entailment
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc['entailment_judgment']
        pred = np.argmax(results)

        return {
            "acc": (pred == gold) * 100.0,
            "f1": (pred, gold),
        }

    @classmethod
    def macro_f1(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        label_set = set(golds)
        macro_f1 = f1_score(golds, preds, average='macro', 
                        labels=list(label_set))
        return macro_f1 * 100.0

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean,
            "f1": self.macro_f1,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            "acc": True,
            "f1": True,
        }


class ASSIN_RTE_GREEDY(ASSIN_RTE):
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
        gold = self.doc_to_target(doc).strip().lower()
        pred = results[0].strip().lower()

        return {
            "acc": (pred == gold) * 100.0,
            "f1": (pred, gold),
        }


_manual_examples=[
    # 1.0
    {'sentence_pair_id': 3288, 'premise': 'Um homem está afagando um jacaré na boca', 'hypothesis': 'Vegetais picados estão sendo despejados dentro de uma panela no fogão por uma pessoa', 'relatedness_score': 1.0, 'entailment_judgment': 0, 'sts_target': 1},
    {'sentence_pair_id': 3316, 'premise': 'Um caminhão está rapidamente subindo um morro', 'hypothesis': 'Um pouco de água está sendo bebida por um gato', 'relatedness_score': 1.0, 'entailment_judgment': 0, 'sts_target': 1},
    {'sentence_pair_id': 3324, 'premise': 'Não tem nenhum homem sentado na cadeira', 'hypothesis': 'A menina, que é pequena, está penteando o cabelo em um rabo de cavalo', 'relatedness_score': 1.0, 'entailment_judgment': 0, 'sts_target': 1},
    # 2.0
    {'sentence_pair_id': 3277, 'premise': 'Um homem está extraindo perigosamente facas de uma árvore', 'hypothesis': 'O homem está fazendo um truque de magia', 'relatedness_score': 2.0, 'entailment_judgment': 0, 'sts_target': 2},
    {'sentence_pair_id': 3610, 'premise': 'Um homem está cuspindo no chão', 'hypothesis': 'Um homem está conversando', 'relatedness_score': 2.0, 'entailment_judgment': 0, 'sts_target': 2},
    {'sentence_pair_id': 3998, 'premise': 'Um menino está tocando um piano', 'hypothesis': 'Alguém está cantando sobre um homem que toca o violão', 'relatedness_score': 2.0, 'entailment_judgment': 0, 'sts_target': 2},
    # 3.0
    {'sentence_pair_id': 772, 'premise': 'A gruta com interior rosa está sendo escalada por quatro crianças do Oriente Médio, três meninas e um menino', 'hypothesis': 'Um grupo de crianças está brincando em uma estrutura colorida', 'relatedness_score': 3.0, 'entailment_judgment': 1, 'sts_target': 3},
    {'sentence_pair_id': 3342, 'premise': 'Uma mulher está fatiando uma batata', 'hypothesis': 'O homem está fatiando vegetais', 'relatedness_score': 3.0, 'entailment_judgment': 0, 'sts_target': 3},
    {'sentence_pair_id': 3344, 'premise': 'Um jogador está correndo com a bola', 'hypothesis': 'Futebol está sendo jogado por dois times concorrentes', 'relatedness_score': 3.0, 'entailment_judgment': 0, 'sts_target': 3},
    # 4.0
    {'sentence_pair_id': 97, 'premise': 'O campo verde para corrida de cavalos está completamente cheio de jóqueis', 'hypothesis': 'Os jóqueis estão correndo a cavalos no campo, que é completamente verde', 'relatedness_score': 4.0, 'entailment_judgment': 1, 'sts_target': 4},
    {'sentence_pair_id': 152, 'premise': 'O praticante de snowboard está saltando sobre da neve branca', 'hypothesis': 'Uma pessoa que está praticando snowboard está pulando no ar', 'relatedness_score': 4.0, 'entailment_judgment': 1, 'sts_target': 4},
    {'sentence_pair_id': 1502, 'premise': 'Dois homens estão lutando boxe', 'hypothesis': 'Dois homens estão lutando', 'relatedness_score': 4.0, 'entailment_judgment': 1, 'sts_target': 4},
    # 5.0
    {'sentence_pair_id': 104, 'premise': 'Um cachorro está saltando em um trampolim', 'hypothesis': 'Um cachorro está pulando em um trampolim', 'relatedness_score': 5.0, 'entailment_judgment': 1, 'sts_target': 5},
    {'sentence_pair_id': 129, 'premise': 'A mulher está cortando cebolas em cubos', 'hypothesis': 'A senhora está cortando cebolas em cubos', 'relatedness_score': 5.0, 'entailment_judgment': 1, 'sts_target': 5},
    {'sentence_pair_id': 131, 'premise': 'Uma mulher está mostrando um cachorro com pelo muito longo em uma exposição canina', 'hypothesis': 'Um cachorro com pelo muito longo está sendo exibido por uma mulher em uma exposição canina', 'relatedness_score': 5.0, 'entailment_judgment': 1, 'sts_target': 5},
]

class ASSIN_STS(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "assin2"
    DATASET_NAME = None

    KEYS_TO_INDEX = ['premise', 'hypothesis']
    KEY_TO_BALANCE = 'sts_target'
    NUM_CLASSES = 5
    SEARCHER_K = 600

    sts_score_choices=[1,2,3,4,5]

    manual_examples=_manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)

        # include sts_target directly after download because it is used
        # to balance the prompts
        for split_data in self.dataset:
            self.dataset[split_data] = list(map(self._process_doc, self.dataset[split_data]))

    def _process_doc(self, doc):
        doc['sts_target'] = round(doc["relatedness_score"])
        return doc

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True
    
    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Sentença 1: {doc['premise']}\nSentença 2: {doc['hypothesis']}\nResposta:"

    def doc_to_target(self, doc):
        return " {}".format(doc["sts_target"])

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
        lls = [
            rf.loglikelihood(ctx, f" {choice}")[0]
            for choice in self.sts_score_choices
        ]
        return lls
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc['relatedness_score']

        pred = float(
            self.sts_score_choices[np.argmax(results)]
        )

        return {
            "mse": (gold - pred)**2,
            "pearson": (pred, gold),
        }

    @classmethod
    def pearson(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        pearson = pearsonr(golds, preds)[0]
        return pearson

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return {
            "mse": mean,
            "pearson": self.pearson,
            "unknown_pred": mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            "mse": False,
            "pearson": True,
            "unknown_pred": False,
        }
    

class ASSIN_STS_GREEDY(ASSIN_STS):
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
        gold = doc['relatedness_score']

        try:
            pred = float(results[0])
            unknown_pred = 0.0
            
        except ValueError as e:
            print(e)
            print("Error parsing answer. Forcing pred=1.0 and gold=5.0 for maximum MSE and lowest Pearson.")
            pred = 1.0
            gold = 5.0
            unknown_pred = 1.0

        debug_info = {
            "gold": gold,
            "pred": pred,
        }

        return {
            "mse": (gold - pred)**2,
            "pearson": (pred, gold),
            "debug_info": debug_info,
            "unknown_pred": unknown_pred,
        }
