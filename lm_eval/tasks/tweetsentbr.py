"""
Building a Sentiment Corpus of Tweets in Brazilian Portuguese
http://www.lrec-conf.org/proceedings/lrec2018/summaries/389.html

TweetSentBR is a sentiment corpus for Brazilian Portuguese manually annotated 
with 15.000 sentences on TV show domain. The sentences were labeled in three 
classes (positive, neutral and negative) by seven annotators, following 
literature guidelines for ensuring reliability on the annotation. 

Homepage: "https://bitbucket.org/HBrum/tweetsentbr/src/master/"
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lm_eval.base import rf, PromptSelectionTask
from ..metrics import mean


_CITATION = """
@InProceedings{BRUM18.389,
  author = {Henrico Brum and Maria das Gra\c{c}as Volpe Nunes},
  title = "{Building a Sentiment Corpus of Tweets in Brazilian Portuguese}",
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {May 7-12, 2018},
  address = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and HÚlŔne Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {979-10-95546-00-9},
  language = {english}
}
"""

_manual_examples=[
    {"id": 862321454427381760, "hashtag": "#encontro", "votes": [1], "hard": 0, "sentiment": 2, "text": "sandy encara o desafio e canta evidência", "repeat": False},
    {"id": 865578874155012096, "hashtag": "#encontro", "votes": [1], "hard": 0, "sentiment": 2, "text": "já quero as irmãs galvão todos os dias nesse programa", "repeat": False},
    {"id": 863032834805690368, "hashtag": "#encontro", "votes": [1], "hard": 0, "sentiment": 2, "text": "ahhh genteee coisa mais linda esse", "repeat": False},
    {"id": 864904757525315584, "hashtag": "#videoShowAoVivo", "votes": [1], "hard": 0, "sentiment": 2, "text": "fatima está cada vez melhor impressionante mais bonita mais simpática oooo mulher inteligente #videoshowaovivo", "repeat": False},
    {"id": 862284663452250112, "hashtag": "#masterchefbr", "votes": [1], "hard": 0, "sentiment": 2, "text": "ontem consegui assistir o que SHOW que emoção", "repeat": False},
    {"id": 864342325647900672, "hashtag": "#ConversaComBial", "votes": [1], "hard": 0, "sentiment": 2, "text": "inteligente charmoso talentoso e ainda cuida da reforma íntima ai ai rodrigo santoro 😍", "repeat": False},
    {"id": 862292396339548160, "hashtag": "#masterchefbr", "votes": [1], "hard": 0, "sentiment": 2, "text": "graças a deus yuco não ficou entre os destaques negativos <3", "repeat": False},
    {"id": 864889834644656128, "hashtag": "#videoShowAoVivo", "votes": [1], "hard": 0, "sentiment": 2, "text": "começou o com a melhor dupla USERNAME e USERNAME ❤", "repeat": False},
    {"id": 864687014486183936, "hashtag": "#ConversaComBial", "votes": [1], "hard": 0, "sentiment": 2, "text": "sensacional a resposta sobre a vaidade #conversacombial", "repeat": False},
    {"id": 864682401854885888, "hashtag": "#ConversaComBial", "votes": [1], "hard": 0, "sentiment": 2, "text": "ahhh como é bom ouvir USERNAME falar sempre confortante", "repeat": False},
    {"id": 865902624947752960, "hashtag": "#édecasa", "votes": [0], "hard": 0, "sentiment": 1, "text": "isso não é granola é uma mistura de cereais #EDeCasa", "repeat": False},
    {"id": 862150150957015040, "hashtag": "#masterchefbr", "votes": [0], "hard": 0, "sentiment": 1, "text": "acho que o victor B nunca teve em tanta pressão na vida inteira", "repeat": False},
    {"id": 861283402523267072, "hashtag": "#DomingoLegal", "votes": [0], "hard": 0, "sentiment": 1, "text": "TCHAU PESSOAL ATÉ DOMINGO QUE VEM ! ! #DomingoLegal", "repeat": False},
    {"id": 863209703278620672, "hashtag": "#masterchefbr", "votes": [0], "hard": 0, "sentiment": 1, "text": "eu só sei fazer miojo", "repeat": False},
    {"id": 865268237050761216, "hashtag": "#videoShowAoVivo", "votes": [0], "hard": 0, "sentiment": 1, "text": "quem canta a música é adriana", "repeat": False},
    {"id": 864817948317413376, "hashtag": "#maisvoce", "votes": [0], "hard": 0, "sentiment": 1, "text": "tem anos que não uso ferro que não passo roupa #MaisVocê", "repeat": False},
    {"id": 863086763224571904, "hashtag": "#videoShowAoVivo", "votes": [0], "hard": 0, "sentiment": 1, "text": "boa tarde eu acho que é o zac de rock story", "repeat": False},
    {"id": 864694948146401280, "hashtag": "#TheNoite", "votes": [0], "hard": 0, "sentiment": 1, "text": "logo após tem USERNAME no comando do #SBTNotícias #NãoPercam #RT !", "repeat": False},
    {"id": 864464101585637376, "hashtag": "#maisvoce", "votes": [0], "hard": 0, "sentiment": 1, "text": "#MaisVocê será q viviane vai recusar o frango", "repeat": False},
    {"id": 863585607402041344, "hashtag": "#altasHoras", "votes": [0], "hard": 0, "sentiment": 1, "text": "sophia abrahão tá cada dia mais parecida com a miley cyrus", "repeat": False},
    {"id": 862152700498255872, "hashtag": "#masterchefbr", "votes": [-1], "hard": 0, "sentiment": 0, "text": "TIRA A MIRIAM CARALHO", "repeat": False},
    {"id": 865725866227228672, "hashtag": "#masterchefbr", "votes": [-1], "hard": 0, "sentiment": 0, "text": "se a yuko sair eu choro 😪", "repeat": False},
    {"id": 865547794915803136, "hashtag": "#maisvoce", "votes": [-1], "hard": 0, "sentiment": 0, "text": "um dos programas mais chatos da TV", "repeat": False},
    {"id": 862140699948535808, "hashtag": "#masterchefbr", "votes": [-1], "hard": 0, "sentiment": 0, "text": "tadinho do vitor B 😢", "repeat": False},
    {"id": 864529418055811072, "hashtag": "#videoShowAoVivo", "votes": [-1], "hard": 0, "sentiment": 0, "text": "sofia tá feinha hoje roupa feia cabelo feio e o batom feioso rsr", "repeat": False},
    {"id": 865675604317806592, "hashtag": "#masterchefbr", "votes": [-1], "hard": 0, "sentiment": 0, "text": "A única participante que me importada e eliminaram foi a A douglas podem eliminar a mentirosa truqueira que é profissional", "repeat": False},
    {"id": 864704595632824320, "hashtag": "#TheNoite", "votes": [-1], "hard": 0, "sentiment": 0, "text": "esquerdistas s contra o porte d arma pois falam q ameaça a vida d outras pessoas mas aborto pode né hipocrisia #thenoite USERNAME", "repeat": False},
    {"id": 864777200377229312, "hashtag": "#masterchefbr", "votes": [-1], "hard": 0, "sentiment": 0, "text": "essa mirian só quer fazer o que dá na telha ô mlr chata", "repeat": False},
    {"id": 862144513137143808, "hashtag": "#masterchefbr", "votes": [-1], "hard": 0, "sentiment": 0, "text": "fiquei com dó da mirian 😢 #MasterchefBR", "repeat": False},
    {"id": 864853986377175040, "hashtag": "#encontro", "votes": [-1], "hard": 0, "sentiment": 0, "text": "A corrupção está espalhada pelo mundo realmente mas daí comparar a nossa corrupção com a do povo japonês já é demais né #ainao", "repeat": False}
]

class TweetSentBR(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "data/tweetsentbr"
    DATASET_NAME = None

    manual_examples = _manual_examples

    KEYS_TO_INDEX = ['text']
    KEY_TO_BALANCE = 'sentiment'
    NUM_CLASSES = 3
    SEARCHER_K = 600

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """ Dowloading the dataset requires the use of an API key. To use the 
        dataset for academic purpose, please contact the corresponding author 
        Henrico Brum. The dataset is originally in TSV format, and comes with 
        a parsing code that converts to a JSON file.
        """
        self.dataset = {}
        # using version of Henrico
        data = pd.read_json(os.path.join(self.DATASET_PATH, 'tweetsentbr.json'), orient='index')

        # remove N/A
        data.dropna(subset=['text'], inplace=True)
        
        # remove cases undefined class (Henrico version)
        data = data[data.sentiment != '-']

        # set labels
        data.sentiment = data.sentiment.astype(int) + 1

        # remove double spaces
        data.text = data.text.apply(lambda x: re.sub(r'(\s)\1*', r' ', x))

        self.dataset['train'] = data.loc[data['group'] == 'train'].drop(columns=['group'])
        self.dataset['test'] = data.loc[data['group'] == 'test'].drop(columns=['group'])

        self.dataset['train'] = self.dataset['train'].to_dict(orient='records')
        self.dataset['test'] = self.dataset['test'].to_dict(orient='records')

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
       return self.dataset["test"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Mensagem: \"{doc['text']}\".\nResposta:"

    def doc_to_target(self, doc):
        return " " + ["negativa", "neutra", "positiva"][doc['sentiment']]

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
        ll_negativa, _ = rf.loglikelihood(ctx, " negativa") 
        ll_neutra, _ = rf.loglikelihood(ctx, " neutra") 
        ll_positiva, _ = rf.loglikelihood(ctx, " positiva") 
        return ll_negativa, ll_neutra, ll_positiva
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc['sentiment']
        pred = np.argmax(results)

        return {
            "acc": (pred == gold) * 100.0,
            "f1-macro": (pred, gold),
            "f1-weighted": (pred, gold),
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

    @classmethod
    def weighted_f1(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        label_set = set(golds)
        weighted_f1 = f1_score(golds, preds, average='weighted', 
                        labels=list(label_set))
        return weighted_f1 * 100.0

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean,
            "f1-macro": self.macro_f1,
            "f1-weighted": self.weighted_f1,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            "acc": True,
            "f1-macro": True,
            "f1-weighted": True,
        }


class TweetSentBR_GREEDY(TweetSentBR):
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
            "f1-macro": (pred, gold),
            "f1-weighted": (pred, gold),
        }