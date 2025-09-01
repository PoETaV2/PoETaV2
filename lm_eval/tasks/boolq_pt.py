"""
BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions
https://arxiv.org/abs/1905.10044
BoolQ is a question answering dataset for yes/no questions containing 15942 examples. These questions are naturally
occurring ---they are generated in unprompted and unconstrained settings.
Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.
The text-pair classification setup is similar to existing natural language inference tasks.
Homepage: https://github.com/google-research-datasets/boolean-questions
"""
import re
from lm_eval.base import rf, PromptSelectionTask
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{clark2019boolq,
  title =     {BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions},
  author =    {Clark, Christopher and Lee, Kenton and Chang, Ming-Wei, and Kwiatkowski, Tom and Collins, Michael, and Toutanova, Kristina},
  booktitle = {NAACL},
  year =      {2019},
}
"""

manual_examples = [
    {'question': 'o discurso do rei ganhou algum oscar', 'answer': True, 'passage': 'No 83º Oscar, o filme recebeu um total de doze indicações ao prêmio, mais do que qualquer outro filme, e venceu quatro: Melhor Filme, Melhor Diretor (Hooper), Melhor Roteiro Original (Seidler) e Melhor Ator (Firth). No 68º Globo de Ouro, o filme recebeu sete indicações, mais do que qualquer outro candidato, mas apenas Firth ganhou um prêmio, de melhor ator. Hooper também ganhou o melhor diretor do 63º Directors Guild of America Awards. No 17º Screen Actors Guild Awards, Firth ganhou o prêmio de Melhor Ator e o elenco ganhou o melhor conjunto. No 64º British Academy Film Awards, ganhou sete prêmios em quatorze indicações, mais do que qualquer outro filme, incluindo melhor filme, melhor filme britânico, Melhor Ator (Firth), Melhor Ator Coadjuvante (Rush), Melhor Atriz Coadjuvante (BonhamCarter), melhor roteiro original (Seidler) e Melhor Música (Alexandre Desplat).'},
    {'question': 'todos os países da América do Sul falam espanhol', 'answer': False, 'passage': 'Português é a língua majoritária da América do Sul, por uma pequena margem. O espanhol, com um pouco menos de falantes nativos que o português, é a segunda língua mais falada do continente.'},
    {'question': 'O rio Mississippi flui para o oceano', 'answer': True, 'passage': "O rio Mississippi possui a quarta maior bacia hidrográfica do mundo. A bacia cobre mais de 3.220.000 km, incluindo todos ou partes de 31 estados dos EUA e duas províncias canadenses. A bacia hidrográfica deságua no Golfo do México, parte do Oceano Atlântico. A captação total do rio Mississippi cobre quase 40% da massa terrestre dos Estados Unidos. O ponto mais alto dentro da bacia hidrográfica também é o ponto mais alto das montanhas rochosas, o Monte Elbert a 14.440 pés (4.400 metros)."},
    {'question': 'Um goleiro pode ser um capitão na NHL', 'answer': False, 'passage': "Antes da temporada de 1948-49, a NHL mudou as regras, proibindo os goleiros de serem capitães ou vice-capitães. Isso foi uma resposta a queixas dos oponentes dos canadenses de Montreal, que reclamaram que Durnan deixou seu vinco para discutir com o árbitro em pontos estratégicos durante os jogos, resultando em tempo limite não programado. Às vezes, essa regra é chamada de `` regra de durnan ''."},
]

class BOOLQPT(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "maritaca-ai/boolq_pt"
    DATASET_NAME = None

    KEYS_TO_INDEX = ['question', 'passage']
    KEY_TO_BALANCE = 'answer'
    NUM_CLASSES = 2
    SEARCHER_K = 10

    manual_examples = manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        # Using docs with less than 749 chars (third quartile),
        # keeping 2388 documents (~75%).
        # np.quantile(lengths, [0.25, 0.5, 0.75]) = array([357. 535. 749.])
        # min(lengths) = 34, max(lengths) = 4971
        self.dataset["train"] = self.dataset["train"].filter(lambda x: len(x["passage"]) < 749)

        def truncate(example):
            # truncation affects 5% of the validation set.
            example["passage"] = example["passage"][:1169]
            return example

        self.dataset["validation"] = self.dataset["validation"].map(truncate)

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
        return f"Pergunta: {doc['question']}\nContexto: {doc['passage']}\nResposta:"

    def doc_to_target(self, doc):
        return " {}".format(["não", "sim"][doc["answer"]])

    def construct_requests(self, doc, ctx):
        ll_negative, _ = rf.loglikelihood(ctx, " não")
        ll_positive, _ = rf.loglikelihood(ctx, " sim")
        return ll_negative, ll_positive

    def process_results(self, doc, results):
        ll_negative, ll_positive = results
        pred = ll_positive > ll_negative
        gold = doc["answer"]
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


class BOOLQPT_GREEDY(BOOLQPT):
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
