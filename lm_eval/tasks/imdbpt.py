"""
Learning Word Vectors for Sentiment Analysis.
https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf

This is a dataset for binary sentiment classification containing substantially 
more data than previous benchmark datasets. We provide a set of 25,000 highly 
polar movie reviews for training, and 25,000 for testing. There is additional 
unlabeled data for use as well. Raw text and already processed bag of words 
formats are provided. See the README file contained in the release for more details.

Homepage: http://ai.stanford.edu/~amaas/data/sentiment/
"""
import re
from lm_eval.base import rf, PromptSelectionTask
from lm_eval.metrics import mean


_CITATION = """
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

manual_examples = [
    {'text': 'O filme de Brooks mais engraçado que eu já vi. Eu posso assistir e assistir novamente a fita 100 vezes. Eu ri meu ** e choro em alguns momentos. É um filme muito bom e engraçado, e se você gosta de Brooks - isso é uma obrigação! Em resumo - Brooks (bilionário) chega às ruas como sem-teto por 30 dias para ganhar todo o pobre distrito de seu concorrente. A realidade morde, mas no final - trata-se de relações calorosas entre os humanos... recomendo!', 'label': 1},
    {'text': "Este não é o pior filme que eu já vi, mas realmente não me lembro quando vi um pior. Eu pensei que isso seria sobre uma investigação de acidentes de aeronaves. O que realmente era é uma novela; e é uma má novela. Eles exageraram com o 'conflito' ao extremo. A primeira hora parece uma partida de gritos, com algumas cenas implausíveis, e eu pensei que nunca terminaria. Evite este filme a todo custo, a menos que você se deleite em 'conflito'.", 'label': 0},
]

class IMDBPT(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "maritaca-ai/imdb_pt"
    DATASET_NAME = None

    KEYS_TO_INDEX = ['text']
    KEY_TO_BALANCE = 'label'
    NUM_CLASSES = 2
    SEARCHER_K = 10

    manual_examples = manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        # Using docs with less than 1544 chars (third quartile, post regex),
        # keeping 18613 documents (~75%).
        # np.quantile(lengths, [0.25, 0.5, 0.75])=array([ 687.  ,  950.  , 1544.25])
        # min(lengths)=44, max(lengths)=13606
        self.dataset["train"] = [
            {
                'text': re.sub('<br />|<BR />|<Br />', ' ', doc['text']),
                'label': doc['label']
            }
            for doc in self.dataset["train"] if len(doc['text']) < 1544
        ]
        # Truncating the test documents that surpasses 2095 chars. This affects 
        # 13% of the test documents (and works near to the limit). 
        # np.quantile(lengths, [0.25, 0.5, 0.75])=array([ 685. ,  937.5, 1515. ])
        # min(lengths)=30, max(lengths)=13217
        self.dataset["test"] = [
            {
                'text': re.sub('<br />|<BR />|<Br />', ' ', doc['text'][:2095]),
                'label': doc['label']
            }
            for doc in self.dataset['test']
        ]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Crítica: {doc['text']}\nAvaliação:"

    def doc_to_target(self, doc):
        return " {}".format({1: "positiva", 0: "negativa"}[doc["label"]])

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


class IMDBPT_GREEDY(IMDBPT):
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
