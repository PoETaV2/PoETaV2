"""
Character-level Convolutional Networks for Text Classification
https://arxiv.org/pdf/1509.01626.pdf

AG is a collection of more than 1 million news articles. News articles have been 
gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of 
activity. ComeToMyHead is an academic news search engine which has been running 
since July, 2004. The dataset is provided by the academic comunity for research 
purposes in data mining (clustering, classification, etc), information retrieval 
(ranking, search, etc), xml, data compression, data streaming, and any other 
non-commercial activity.

ag_news_pt is the Portuguese translation of the ag_news dataset, using Google 
Cloud Translation.

Homepage: "http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html"
"""
import numpy as np
import re
from sklearn.metrics import f1_score
from lm_eval.base import rf, PromptSelectionTask
from ..metrics import mean


_CITATION = """
@inproceedings{Zhang2015CharacterlevelCN,
    title={Character-level Convolutional Networks for Text Classification},
    author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
    booktitle={NIPS},
    year={2015}
}
"""

manual_examples = [
    {'title': 'Vendas de residências existentes disparam em setembro', 'text': 'WASHINGTON (Reuters) - As vendas de residências existentes nos Estados Unidos subiram inesperadamente 3,1 por cento em setembro devido às baixas taxas de hipoteca, e teriam sido ainda mais intensas se os furacões não tivessem atingido o sul, disse uma associação comercial nesta segunda-feira.', 'label': 2},
    {'title': 'Corpos de dois ocidentais encontrados ao sul de Bagdá (Reuters)', 'text': 'Reuters - A polícia iraquiana encontrou os corpos de um homem e uma mulher, ambos que se acredita serem ocidentais, ao sul de Bagdá, disse o diretor do hospital que recebeu os cadáveres no domingo.', 'label': 0},
    {'title': 'Marte: Formas de polígono podem ser resultado da ação da água', 'text': 'As curiosas formas de polígono na superfície de Marte estão entre as últimas evidências sugerindo claramente a presença de água, e algumas delas podem ter aparecido lá mesmo depois que a superfície foi bombardeada por objetos do espaço distante.', 'label': 3},
    {'title': 'Mais dois americanos não conseguem dominar o estilo de boxe olímpico', 'text': 'Sem chance, pai. Os melhores boxeadores amadores sabem como se mover e exatamente quando atacar - e três americanos que não conseguiram igualar esse estilo estão fora das Olimpíadas.', 'label': 1},
    {'title': 'Forças de segurança em alerta antes da eleição afegã', 'text': 'Mais de 100.000 forças de segurança afegãs e estrangeiras ficaram em alerta na sexta-feira, um dia antes da primeira eleição presidencial direta do país após mais de duas décadas de guerra.', 'label': 0},
    {'title': 'A sonda de Saturno pode revelar a chave da vida', 'text': 'Uma investigação espacial foi a maior lua de Saturno Tita e está chegando ao final de sua jornada de dois bilhões de milhas, relata Nic Fleming.', 'label': 3},
    {'title': 'A contagem regressiva começa', 'text': 'Quarta-feira foi um grande dia na contagem regressiva para o Super Bowl 2006, quando a NFL chegou a Dearborn na quarta-feira para revelar o logotipo oficial do Big Event de Detroits.', 'label': 1},
    {'title': 'Equipe de Utah contrata Ellinger como técnico (AP)', 'text': 'AP - John Ellinger, técnico da seleção sub-17 dos EUA nos últimos sete anos, foi contratado na quarta-feira como técnico da equipe de expansão da Major League Soccer em Utah (Principal Liga de Futebol de Utah).', 'label': 1},
    {'title': 'EUA se opõem às negociações com seqüestradores afegãos: Armitage (AFP)', 'text': 'AFP - Os Estados Unidos se opõem às negociações com os sequestradores que sequestraram três funcionários das Nações Unidas no Afeganistão no mês passado, disse o vice-secretário de Estado dos EUA, Richard Armitage.', 'label': 0},
    {'title': 'Não pare de negociar', 'text': 'Tem sido um namoro longo e torturado. Durante anos, a gigante da saúde Johnson & Johnson (JNJ) manteve conversas de aquisição de novo e de novo com o fabricante de dispositivos médicos Guidant (GDT). Agora, as fontes de Wall Street dizem que as empresas estão conversando novamente.', 'label': 2},
    {'title': 'Câmara dos Deputados dos EUA aprova lei mais dura contra o comércio de arquivos', 'text': 'SAN FRANCISCO - A Câmara dos Deputados dos Estados Unidos aprovou na terça-feira um projeto de lei que pode permitir que acusações criminais sejam feitas contra indivíduos que participam de sites ou redes de troca de arquivos.', 'label': 3},
    {'title': 'Após a campainha: as ações da Synopsys caem', 'text': 'NOVA YORK (Reuters) - Ações da Synopsys Inc. caía na quarta-feira, depois que a fabricante de software de design de semicondutores relatou um lucro líquido mais baixo no terceiro trimestre e disse que os ganhos e receitas no quarto trimestre estariam abaixo das estimativas de Wall Street.', 'label': 2},
]

class AGNewsPT(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "maritaca-ai/ag_news_pt"
    DATASET_NAME = None
    
    KEYS_TO_INDEX = ['text']
    KEY_TO_BALANCE = 'label'
    NUM_CLASSES = 4
    SEARCHER_K = 300

    manual_examples = manual_examples

    topics = {
        0: 'Mundo',
        1: 'Esportes',
        2: 'Negócios',
        3: 'Tecnologia',
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)
        # Removing links and some html artifacts.
        # Using docs with less than 298 chars (third quartile), keeping 88454 documents (73.71%)
        self.dataset["train"] = [
            doc for doc in self.dataset["train"] if not re.search(r'http|lt\s*;|gt\s*;', doc['text']) 
            and len(doc['text']) < 298
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
        return f"Título: {doc['title']}\nTexto: {doc['text']}\nCategoria:"

    def doc_to_target(self, doc):
        return " " + self.topics[doc['label']]

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
        likelihoods = []
        for topic in self.topics.values():
            ll, _ = rf.loglikelihood(ctx, " " + topic)
            likelihoods.append(ll)
        return likelihoods
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc['label']
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


class AGNewsPT_GREEDY(AGNewsPT):
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
