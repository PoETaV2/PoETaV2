"""
Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
https://aclanthology.org/D13-1170/
The Stanford Sentiment Treebank consists of sentences from movie reviews and
human annotations of their sentiment. The task is to predict the sentiment of a
given sentence. We use the two-way (positive/negative) class split, and use only
sentence-level labels.
Homepage: https://nlp.stanford.edu/sentiment/
"""
import re
from lm_eval.base import rf, PromptSelectionTask
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
"""

manual_examples = [
    {'text': 'desperdiçado quase duas horas da sua própria vida preciosa', 'label': 0},
    {'text': 'são simplesmente bons demais', 'label': 1},
    {'text': 'O talento meticuloso de Tian', 'label': 1},
    {'text': 'sobrecarregado com plot complicado e diálogo banal', 'label': 0},
    {'text': 'genuinamente engraçado', 'label': 1},
    {'text': '"MIB II" é bem-sucedido devido à sua entrega rápida e à leviandade inspirada o suficiente para que não seja demitido como irracional.', 'label': 1},
    {'text': 'confunde sua mensagem com um desejo final de agradar e se contorcer em uma idéia de expectativa', 'label': 0},
    {'text': 'deliciosamente encantador', 'label': 1},
    {'text': 'tem a sensação de um show de talentos de acampamento de verão: escrito às pressas', 'label': 0},
    {'text': 'humor idiota', 'label': 0},
    {'text': 'preocupado em construir uma história real desta vez', 'label': 1},
    {'text': 'Um filme de desastre dos anos 70 mundano', 'label': 0},
    {'text': 'Um filme de vaidade assustador que, sem dúvida, paga o que a dívida que Miramax devia a Benigni', 'label': 1},
    {'text': 'Tanto uma adaptação bem-sucedida quanto um filme agradável por si só.', 'label': 1},
    {'text': 'Um enorme sucesso de bilheteria na Coréia, Shiri é uma obrigação para os fãs de gênero.', 'label': 1},
    {'text': 'O trabalho de sangue bem feito, mas não envolvido, não é um filme terrível, apenas um óbvio estultificadamente - um colarinho não recompensador para um mistério de assassinato.', 'label': 0},
    {'text': 'é mais do que um filme', 'label': 1},
    {'text': 'Tão satisfatoriamente estranho e intrigante uma história como era um século e meio atrás', 'label': 1},
    {'text': 'não tem um ponto de vista forte, ou um senso de humor', 'label': 0},
    {'text': 'um filme inegavelmente comovente para experimentar', 'label': 1},
    {'text': 'supostamente “baseado em fatos reais”, uma convolução de linguagem que sugere que é impossível afirmar que é “baseado em uma história real” com uma cara séria.', 'label': 0},
    {'text': 'quem se importa?', 'label': 0},
    {'text': 'Um filme em uma aula com a coisa mais importante de Spike Lee.', 'label': 1},
    {'text': 'começa como um filme infantil satisfatório que se torna cada vez mais implausível à medida que corre através de pontos de enredo planejados', 'label': 0},
    {'text': 'seu enredo banal e mesquinho', 'label': 0},
    {'text': 'brilha como um farol solitário.', 'label': 1},
    {'text': 'menos engraçado do que deveria ser e menos engraçado do que pensa.', 'label': 0},
    {'text': 'provavelmente teria funcionado melhor como um documentário de TV de uma hora', 'label': 0},
    {'text': 'Documentário de Hollywood estranhamente honesto', 'label': 1},
    {'text': 'é uma contemplação visualmente impressionante sobre amor, memória, história e a guerra entre arte e comércio.', 'label': 1},
    {'text': 'um drama humano maravilhosamente caloroso que permanece vívido na memória por muito tempo', 'label': 1},
    {'text': 'filme de assalto sem graça que começam a evaporar da sua memória minutos depois de terminar.', 'label': 0},
    {'text': 'isso é trágico demais para merecer um tratamento tão superficial', 'label': 0},
    {'text': 'é repleto de simbolismo forçado, psicologia barata e intermináveis tomadas cênicas que fazem 105 minutos parecerem o dobro do tempo.', 'label': 0},
]


class SST2PT(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "maritaca-ai/sst2_pt"
    DATASET_NAME = None

    KEYS_TO_INDEX = ['text']
    KEY_TO_BALANCE = 'label'
    NUM_CLASSES = 2
    SEARCHER_K = 10

    manual_examples = manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        # Using docs with less than 75 chars (third quartile),
        # keeping 50182 documents (~75%).
        # np.quantile(lengths, [0.25, 0.5, 0.75]) = array([21. 40. 75.])
        # min(lengths) = 1, max(lengths) = 285
        # self.dataset["train"] = self.dataset["train"].filter(lambda x: len(x["text"]) < 75)

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
        return f"Crítica: {doc['text']}\nAvaliação:"

    def doc_to_target(self, doc):
        return " {}".format(["negativa", "positiva"][doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_negative, _ = rf.loglikelihood(ctx, " negativa")
        ll_positive, _ = rf.loglikelihood(ctx, " positiva")
        return ll_negative, ll_positive

    def process_results(self, doc, results):
        ll_negative, ll_positive = results
        pred = ll_positive > ll_negative
        gold = doc["label"]
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }


class SST2PT_GREEDY(SST2PT):
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
            "acc": (pred == gold),
        }
