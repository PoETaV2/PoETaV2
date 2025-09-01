"""
The Winograd Schema Challenge
http://commonsensereasoning.org/2011/papers/Levesque.pdf

A Winograd schema is a pair of sentences that differ in only one or two words
and that contain an ambiguity that is resolved in opposite ways in the two
sentences and requires the use of world knowledge and reasoning for its resolution.
The Winograd Schema Challenge 285 is a collection of 285 such Winograd schemas.

NOTE: This evaluation of Winograd Schema Challenge is based on `partial evaluation`
as described by Trinh & Le in Simple Method for Commonsense Reasoning (2018).
See: https://arxiv.org/abs/1806.0

NOTE: This evaluation utilizes a collection of Winograd schemas that are based 
on the Portuguese language. Specifically, the version of the dataset used 
includes translations of the most commonly found names into Portuguese. 
Additionally, the evaluation process focuses on the sentences labeled as 
`manually_fixed_correct_sentence` and `manually_fixed_incorrect_sentence`, 
which respectively correspond to the versions of the sentences that are 
either correct or incorrect, both with manual fixings.

Homepages: 
https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html
https://github.com/gabimelo/portuguese_wsc
"""
import copy
import json
import re
import numpy as np
import os
from lm_eval.base import rf, PromptSelectionTask
from lm_eval.metrics import mean
from urllib.request import urlretrieve


_CITATION = """
@inproceedings{ea01b9c0db064caca6986b925d75f2bb,
    title = "The winograd schema challenge",
    abstract = "In this paper, we present an alternative to the Turing Test that has some conceptual and practical advantages. A Wino-grad schema is a pair of sentences that differ only in one or two words and that contain a referential ambiguity that is resolved in opposite directions in the two sentences. We have compiled a collection of Winograd schemas, designed so that the correct answer is obvious to the human reader, but cannot easily be found using selectional restrictions or statistical techniques over text corpora. A contestant in the Winograd Schema Challenge is presented with a collection of one sentence from each pair, and required to achieve human-level accuracy in choosing the correct disambiguation.",
    author = "Levesque, {Hector J.} and Ernest Davis and Leora Morgenstern",
    year = "2012",
    language = "English (US)",
    isbn = "9781577355601",
    series = "Proceedings of the International Conference on Knowledge Representation and Reasoning",
    publisher = "Institute of Electrical and Electronics Engineers Inc.",
    pages = "552--561",
    booktitle = "13th International Conference on the Principles of Knowledge Representation and Reasoning, KR 2012",
    note = "13th International Conference on the Principles of Knowledge Representation and Reasoning, KR 2012 ; Conference date: 10-06-2012 Through 14-06-2012",
}
@inproceedings{eniac,
    author = {Gabriela Melo and Vinicius Imaizumi and Fábio Cozman},
    title = {Winograd Schemas in Portuguese},
    booktitle = {Anais do XVI Encontro Nacional de Inteligência Artificial e Computacional},
    location = {Salvador},
    year = {2019},
    keywords = {},
    issn = {2763-9061},
    pages = {787--798},
    publisher = {SBC},
    address = {Porto Alegre, RS, Brasil},
    doi = {10.5753/eniac.2019.9334},
    url = {https://sol.sbc.org.br/index.php/eniac/article/view/9334}
}
"""

_manual_examples=[
  {
    "question_id": 2,
    "correct_sentence": "A medalha não cabe na maleta porque a medalha é muito grande.",
    "incorrect_sentence": "A medalha não cabe na maleta porque a maleta é muito grande.",
    "manually_fixed_correct_sentence": "A medalha não cabe na maleta porque a medalha é muito grande.",
    "manually_fixed_incorrect_sentence": "A medalha não cabe na maleta porque a maleta é muito grande.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 9,
    "correct_sentence": "O advogado fez uma pergunta ao acusado, mas o acusado estava relutante em respondê-la.",
    "incorrect_sentence": "O advogado fez uma pergunta ao acusado, mas o advogado estava relutante em respondê-la.",
    "manually_fixed_correct_sentence": "O advogado fez uma pergunta ao acusado, mas o acusado estava relutante em respondê-la.",
    "manually_fixed_incorrect_sentence": "O advogado fez uma pergunta ao acusado, mas o advogado estava relutante em respondê-la.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 17,
    "correct_sentence": "A grande bola atravessou a mesa pois a mesa era feita de isopor.",
    "incorrect_sentence": "A grande bola atravessou a mesa pois a grande bola era feita de isopor.",
    "manually_fixed_correct_sentence": "A grande bola atravessou a mesa pois a mesa era feita de isopor.",
    "manually_fixed_incorrect_sentence": "A grande bola atravessou a mesa pois a grande bola era feita de isopor.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 23,
    "correct_sentence": "Apesar de ambas correrem aproximadamente na mesma velocidade, Sandra derrotou Marcia porque Marcia começou muito mal.",
    "incorrect_sentence": "Apesar de ambas correrem aproximadamente na mesma velocidade, Sandra derrotou Marcia porque Sandra começou muito mal.",
    "manually_fixed_correct_sentence": "Apesar de ambas correrem aproximadamente na mesma velocidade, Sandra derrotou Marcia porque Marcia começou muito mal.",
    "manually_fixed_incorrect_sentence": "Apesar de ambas correrem aproximadamente na mesma velocidade, Sandra derrotou Marcia porque Sandra começou muito mal.",
    "correct_switched": "Apesar de ambas correrem aproximadamente na mesma velocidade, Marcia derrotou Sandra porque Sandra começou muito mal.",
    "incorrect_switched": "Apesar de ambas correrem aproximadamente na mesma velocidade, Marcia derrotou Sandra porque Marcia começou muito mal.",
    "is_associative": False,
    "is_switchable": True,
    "translated": True
  },
  {
    "question_id": 44,
    "correct_sentence": "Vanessa sabe tudo sobre os problemas pessoais da Adriana porque Vanessa é intrometida.",
    "incorrect_sentence": "Vanessa sabe tudo sobre os problemas pessoais da Adriana porque Adriana é intrometida.",
    "manually_fixed_correct_sentence": "Vanessa sabe tudo sobre os problemas pessoais da Adriana porque Vanessa é intrometida.",
    "manually_fixed_incorrect_sentence": "Vanessa sabe tudo sobre os problemas pessoais da Adriana porque Adriana é intrometida.",
    "correct_switched": "Adriana sabe tudo sobre os problemas pessoais da Vanessa porque Adriana é intrometida.",
    "incorrect_switched": "Adriana sabe tudo sobre os problemas pessoais da Vanessa porque Vanessa é intrometida.",
    "is_associative": False,
    "is_switchable": True,
    "translated": True
  },
  {
    "question_id": 51,
    "correct_sentence": "O tio do Jorge ainda consegue derrotar ele no tênis, apesar de o tio do Jorge ser 30 anos mais velho.",
    "incorrect_sentence": "O tio do Jorge ainda consegue derrotar ele no tênis, apesar de Jorge ser 30 anos mais velho.",
    "manually_fixed_correct_sentence": "O tio do Jorge ainda consegue derrotar ele no tênis, apesar de o tio do Jorge ser 30 anos mais velho.",
    "manually_fixed_incorrect_sentence": "O tio do Jorge ainda consegue derrotar ele no tênis, apesar de Jorge ser 30 anos mais velho.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 52,
    "correct_sentence": "O quadro na sala de estar do Marcos mostra um carvalho. O quadro está à direita da estante de livros.",
    "incorrect_sentence": "O quadro na sala de estar do Marcos mostra um carvalho. O carvalho está à direita da estante de livros.",
    "manually_fixed_correct_sentence": "O quadro na sala de estar do Marcos mostra um carvalho. O quadro está à direita da estante de livros.",
    "manually_fixed_incorrect_sentence": "O quadro na sala de estar do Marcos mostra um carvalho. O carvalho está à direita da estante de livros.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 59,
    "correct_sentence": "Meu compromisso começava às 4:00 e eu precisava pegar o trem às 4:30, portanto não havia muito tempo. Felizmente, o trem atrasou, então tudo deu certo.",
    "incorrect_sentence": "Meu compromisso começava às 4:00 e eu precisava pegar o trem às 4:30, portanto não havia muito tempo. Felizmente, meu compromisso atrasou, então tudo deu certo.",
    "manually_fixed_correct_sentence": "Meu compromisso começava às 4:00 e eu precisava pegar o trem às 4:30, portanto não havia muito tempo. Felizmente, o trem atrasou, então tudo deu certo.",
    "manually_fixed_incorrect_sentence": "Meu compromisso começava às 4:00 e eu precisava pegar o trem às 4:30, portanto não havia muito tempo. Felizmente, meu compromisso atrasou, então tudo deu certo.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 65,
    "correct_sentence": "Durante a apresentação a céu aberto, começou a chover, e a chuva continuou até às 10.",
    "incorrect_sentence": "Durante a apresentação a céu aberto, começou a chover, e a apresentação continuou até às 10.",
    "manually_fixed_correct_sentence": "Durante a apresentação a céu aberto, começou a chover, e a chuva continuou até às 10.",
    "manually_fixed_incorrect_sentence": "Durante a apresentação a céu aberto, começou a chover, e a apresentação continuou até às 10.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 66,
    "correct_sentence": "Eu usei um pano velho para limpar o alicate, e então joguei o pano no lixo.",
    "incorrect_sentence": "Eu usei um pano velho para limpar o alicate, e então joguei o alicate no lixo.",
    "manually_fixed_correct_sentence": "Eu usei um pano velho para limpar o alicate, e então joguei o pano no lixo.",
    "manually_fixed_incorrect_sentence": "Eu usei um pano velho para limpar o alicate, e então joguei o alicate no lixo.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 75,
    "correct_sentence": "Tenho certeza que meu mapa vai mostrar esse prédio; o prédio é muito famoso.",
    "incorrect_sentence": "Tenho certeza que meu mapa vai mostrar esse prédio; o mapa é muito famoso.",
    "manually_fixed_correct_sentence": "Tenho certeza que meu mapa vai mostrar esse prédio; o prédio é muito famoso.",
    "manually_fixed_incorrect_sentence": "Tenho certeza que meu mapa vai mostrar esse prédio; o mapa é muito famoso.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": True,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 79,
    "correct_sentence": "Bruno custeou o ensino superior de Fabio, mas agora Fabio age como se isso nunca tivesse acontecido. Fabio é muito ingrato.",
    "incorrect_sentence": "Bruno custeou o ensino superior de Fabio, mas agora Fabio age como se isso nunca tivesse acontecido. Bruno é muito ingrato.",
    "manually_fixed_correct_sentence": "Bruno custeou o ensino superior de Fabio, mas agora Fabio age como se isso nunca tivesse acontecido. Fabio é muito ingrato.",
    "manually_fixed_incorrect_sentence": "Bruno custeou o ensino superior de Fabio, mas agora Fabio age como se isso nunca tivesse acontecido. Bruno é muito ingrato.",
    "correct_switched": "Fabio custeou o ensino superior de Bruno, mas agora Bruno age como se isso nunca tivesse acontecido. Bruno é muito ingrato.",
    "incorrect_switched": "Fabio custeou o ensino superior de Bruno, mas agora Bruno age como se isso nunca tivesse acontecido. Fabio é muito ingrato.",
    "is_associative": False,
    "is_switchable": True,
    "translated": True
  },
  {
    "question_id": 80,
    "correct_sentence": "Bruno estava jogando cartas com Antonio e estava muito na frente no placar. Se Antonio não tivesse tido uma sequência de jogadas de sorte, Bruno teria ganhado.",
    "incorrect_sentence": "Bruno estava jogando cartas com Antonio e estava muito na frente no placar. Se Antonio não tivesse tido uma sequência de jogadas de sorte, Antonio teria ganhado.",
    "manually_fixed_correct_sentence": "Bruno estava jogando cartas com Antonio e estava muito na frente no placar. Se Antonio não tivesse tido uma sequência de jogadas de sorte, Bruno teria ganhado.",
    "manually_fixed_incorrect_sentence": "Bruno estava jogando cartas com Antonio e estava muito na frente no placar. Se Antonio não tivesse tido uma sequência de jogadas de sorte, Antonio teria ganhado.",
    "correct_switched": "Antonio estava jogando cartas com Bruno e estava muito na frente no placar. Se Bruno não tivesse tido uma sequência de jogadas de sorte, Antonio teria ganhado.",
    "incorrect_switched": "Antonio estava jogando cartas com Bruno e estava muito na frente no placar. Se Bruno não tivesse tido uma sequência de jogadas de sorte, Bruno teria ganhado.",
    "is_associative": False,
    "is_switchable": True,
    "translated": True
  },
  {
    "question_id": 94,
    "correct_sentence": "Eu vi o Guilherme gritando com um cara que vestia uma farda e tinha uma grande barba ruiva. Eu não sei porque Guilherme estava assim, mas ele parecia muito chateado.",
    "incorrect_sentence": "Eu vi o Guilherme gritando com um cara que vestia uma farda e tinha uma grande barba ruiva. Eu não sei porque o cara que vestia uma farda estava assim, mas ele parecia muito chateado.",
    "manually_fixed_correct_sentence": "Eu vi o Guilherme gritando com um cara que vestia uma farda e tinha uma grande barba ruiva. Eu não sei porque Guilherme estava assim, mas ele parecia muito chateado.",
    "manually_fixed_incorrect_sentence": "Eu vi o Guilherme gritando com um cara que vestia uma farda e tinha uma grande barba ruiva. Eu não sei porque o cara que vestia uma farda estava assim, mas ele parecia muito chateado.",
    "correct_switched": "Eu vi um cara que vestia uma farda e tinha uma grande barba ruiva gritando com o Guilherme. Eu não sei porque o cara que vestia uma farda estava assim, mas ele parecia muito chateado.",
    "incorrect_switched": "Eu vi um cara que vestia uma farda e tinha uma grande barba ruiva gritando com o Guilherme. Eu não sei porque Guilherme estava assim, mas ele parecia muito chateado.",
    "is_associative": False,
    "is_switchable": True,
    "translated": True
  },
  {
    "question_id": 99,
    "correct_sentence": "Eu estava tentando abrir o cadeado com a chave, mas alguém havia preenchido a fechadura com goma de mascar, e eu não conseguia remover a goma de mascar.",
    "incorrect_sentence": "Eu estava tentando abrir o cadeado com a chave, mas alguém havia preenchido a fechadura com goma de mascar, e eu não conseguia remover a chave.",
    "manually_fixed_correct_sentence": "Eu estava tentando abrir o cadeado com a chave, mas alguém havia preenchido a fechadura com goma de mascar, e eu não conseguia remover a goma de mascar.",
    "manually_fixed_incorrect_sentence": "Eu estava tentando abrir o cadeado com a chave, mas alguém havia preenchido a fechadura com goma de mascar, e eu não conseguia remover a chave.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 105,
    "correct_sentence": "O cliente entrou no banco e esfaqueou um dos caixas. O caixa foi levado imediatamente para o hospital.",
    "incorrect_sentence": "O cliente entrou no banco e esfaqueou um dos caixas. O cliente foi levado imediatamente para o hospital.",
    "manually_fixed_correct_sentence": "O cliente entrou no banco e esfaqueou um dos caixas. O caixa foi levado imediatamente para o hospital.",
    "manually_fixed_incorrect_sentence": "O cliente entrou no banco e esfaqueou um dos caixas. O cliente foi levado imediatamente para o hospital.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 109,
    "correct_sentence": "João estava correndo pelo parque quando viu um homem fazendo malabarismos com melancias. O malabarista era muito impressionante.",
    "incorrect_sentence": "João estava correndo pelo parque quando viu um homem fazendo malabarismos com melancias. João era muito impressionante.",
    "manually_fixed_correct_sentence": "João estava correndo pelo parque quando viu um homem fazendo malabarismos com melancias. O malabarista era muito impressionante.",
    "manually_fixed_incorrect_sentence": "João estava correndo pelo parque quando viu um homem fazendo malabarismos com melancias. João era muito impressionante.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": True,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 114,
    "correct_sentence": "Marcos contou muitas mentiras sobre ele mesmo para o Pedro, e Pedro incluiu elas no seu livro. Marcos deveria ter sido mais sincero.",
    "incorrect_sentence": "Marcos contou muitas mentiras sobre ele mesmo para o Pedro, e Pedro incluiu elas no seu livro. Pedro deveria ter sido mais sincero.",
    "manually_fixed_correct_sentence": "Marcos contou muitas mentiras sobre ele mesmo para o Pedro, e Pedro incluiu elas no seu livro. Marcos deveria ter sido mais sincero.",
    "manually_fixed_incorrect_sentence": "Marcos contou muitas mentiras sobre ele mesmo para o Pedro, e Pedro incluiu elas no seu livro. Pedro deveria ter sido mais sincero.",
    "correct_switched": "Pedro contou muitas mentiras sobre ele mesmo para o Marcos, e Marcos incluiu elas no seu livro. Pedro deveria ter sido mais sincero.",
    "incorrect_switched": "Pedro contou muitas mentiras sobre ele mesmo para o Marcos, e Marcos incluiu elas no seu livro. Marcos deveria ter sido mais sincero.",
    "is_associative": False,
    "is_switchable": True,
    "translated": True
  },
  {
    "question_id": 121,
    "correct_sentence": "Maria pegou sua flauta e tocou uma de suas peças favoritas. Ela ama a peça desde criança.",
    "incorrect_sentence": "Maria pegou sua flauta e tocou uma de suas peças favoritas. Ela ama a flauta desde criança.",
    "manually_fixed_correct_sentence": "Maria pegou sua flauta e tocou uma de suas peças favoritas. Ela ama a peça desde criança.",
    "manually_fixed_incorrect_sentence": "Maria pegou sua flauta e tocou uma de suas peças favoritas. Ela ama a flauta desde criança.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 127,
    "correct_sentence": "Sara pegou um livro emprestado da biblioteca porque ela precisa dele para um artigo no qual ela está trabalhando. Ela escreve o artigo quando chega do trabalho.",
    "incorrect_sentence": "Sara pegou um livro emprestado da biblioteca porque ela precisa dele para um artigo no qual ela está trabalhando. Ela escreve o livro quando chega do trabalho.",
    "manually_fixed_correct_sentence": "Sara pegou um livro emprestado da biblioteca porque ela precisa dele para um artigo no qual ela está trabalhando. Ela escreve o artigo quando chega do trabalho.",
    "manually_fixed_incorrect_sentence": "Sara pegou um livro emprestado da biblioteca porque ela precisa dele para um artigo no qual ela está trabalhando. Ela escreve o livro quando chega do trabalho.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
  {
    "question_id": 128,
    "correct_sentence": "Nesta manhã, José construiu um castelo de areia na praia, e colocou um soldado de brinquedo na torre mais alta, mas nesta tarde a maré derrubou o castelo de areia.",
    "incorrect_sentence": "Nesta manhã, José construiu um castelo de areia na praia, e colocou um soldado de brinquedo na torre mais alta, mas nesta tarde a maré derrubou o soldado.",
    "manually_fixed_correct_sentence": "Nesta manhã, José construiu um castelo de areia na praia, e colocou um soldado de brinquedo na torre mais alta, mas nesta tarde a maré derrubou o castelo de areia.",
    "manually_fixed_incorrect_sentence": "Nesta manhã, José construiu um castelo de areia na praia, e colocou um soldado de brinquedo na torre mais alta, mas nesta tarde a maré derrubou o soldado.",
    "correct_switched": "",
    "incorrect_switched": "",
    "is_associative": False,
    "is_switchable": False,
    "translated": True
  },
]

class WinogradSchemaChallenge285(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "data/wsc285_pt"

    KEYS_TO_INDEX = ["manually_fixed_correct_sentence"]
    
    manual_examples = _manual_examples

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        if os.path.exists(self.DATASET_PATH):
            print(f"Reusing dataset wsc285_pt ({self.DATASET_PATH})")
        else:
            os.makedirs(self.DATASET_PATH, exist_ok=True)
            urlretrieve('https://raw.githubusercontent.com/gabimelo/portuguese_wsc/master/data/processed/portuguese_wsc_portuguese_names.json',
                        os.path.join(self.DATASET_PATH, 'portuguese_wsc_portuguese_names.json'))

        self.dataset = {}
        fname = os.path.join(self.DATASET_PATH, 'portuguese_wsc_portuguese_names.json')
        with open(fname, encoding="utf-8") as f:
            self.dataset['test'] = json.load(f)
        
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)
        # set and process the manual examples
        self.manual_examples = list(map(self._load_doc, self.manual_examples))

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return list(map(self._load_doc, self.dataset["test"]))

    def test_docs(self):
        return list(map(self._load_doc, self.dataset["test"]))

    def _load_doc(self, doc):
        """Adds in the document the position from which the two sentences diverge.
        That position is used to separate text and target.

        :param doc:
            The document with the new argument `position`
        """
        position = 0
        for tok1, tok2 in zip(doc["manually_fixed_correct_sentence"].split(),
                              doc["manually_fixed_incorrect_sentence"].split()):
            if tok1 == tok2:
                position += len(tok1) + 1
            else:
                break
        doc['position'] = max(0, position - 1)
        return doc

    def doc_to_text(self, doc):
        return doc["manually_fixed_correct_sentence"][:doc["position"]]

    def doc_to_target(self, doc):
        return doc["manually_fixed_correct_sentence"][doc["position"]:]

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
        lls = [
            rf.loglikelihood(ctx, doc["manually_fixed_correct_sentence"][doc["position"]:])[0], 
            rf.loglikelihood(ctx, doc["manually_fixed_incorrect_sentence"][doc["position"]:])[0]
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
        # target is 0 because the request of the correct sentence cames first
        target = 0

        return {
            "acc": np.argmax(results) == target
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "acc": True
        }

    def fewshot_examples(self, k, rnd, prompt_mode, doc):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        if prompt_mode == 'dynamic-random':
            # Ignore the current document
            _training_docs = copy.copy(self.training_docs())
            _training_docs.remove(doc)
            return rnd.sample(_training_docs, k)

        elif prompt_mode == 'fixed':
            # Ignore the current document
            _training_docs = copy.copy(self.training_docs())
            _training_docs.remove(doc)
            return rnd.sample(_training_docs[:k], k)

        elif prompt_mode == 'manual':
            assert k <= len(self.manual_examples), (
                f'The number of manual_examples is not enough to satisfy '
                f'num_fewshot={k}. Please, include more examples.')
            # Ignore the current document
            _manual_examples = copy.copy(self.manual_examples)
            if doc in _manual_examples:
                _manual_examples.remove(doc)
            return rnd.sample(_manual_examples, k)

        elif prompt_mode == 'dynamic-similar':
            if self.searcher is None:
                from pyserini.search.lucene import LuceneSearcher
                
                self.indexes_dir = os.path.join('data', self.DATASET_PATH, 'indexes')
                self.documents_to_index = os.path.join(
                    'data', self.DATASET_PATH, 'docs_to_index', 'documents.json')

                # Index the training documents for dynamic-similar prompt mode.
                # Run only once.
                if not os.path.exists(self.indexes_dir):
                    os.makedirs(os.path.dirname(self.documents_to_index), exist_ok=True)
                    self.create_collection_to_index()
                    self.indexing()

                # Instantiate the searcher for dynamic-similar prompts.
                self.searcher = LuceneSearcher(self.indexes_dir)
                self.searcher.set_language('pt')

            hits = self.searcher.search(
                '. '.join(doc[key] for key in self.KEYS_TO_INDEX),
                k=self.SEARCHER_K
            )
            indices = []

            for hit in hits:
                hit = json.loads(hit.raw)

                # Ignore the current document
                if hit['id'] == doc['question_id']:
                    continue
                
                id = hit['id']
                indices.append(id)

                doc_x = self._training_docs[id]
                assert hit['contents'] == '. '.join(doc_x[key] for key in self.KEYS_TO_INDEX)

                if len(indices) == k:
                    break

            # check if each class has enough similar examples. If not, complete 
            # with random examples.
            i = 0
            while len(indices) < k:
                if i not in indices:
                    indices.append(i)
                i+=1
            
            # Move the most relevant examples to the end.
            indices.reverse() 

            return [ self._training_docs[i] for i in indices[:k] ]

        else:
            print('Please set prompt_mode as "fixed", "dynamic-random", "dynamic-similar", or "manual"')


class WinogradSchemaChallenge285_GREEDY(WinogradSchemaChallenge285):
    """Additional implementation that uses multiple-choice-like prompts and 
    works with greedy-until requests. This task can be adapted to use 
    loglikelihood-requests, while the original task cannot.
    """
    
    def doc_to_text(self, doc):
        if doc["question_id"] % 2 == 0:
            return (f'{doc["manually_fixed_correct_sentence"][:doc["position"]]}\n'
                    f'A.{doc["manually_fixed_correct_sentence"][doc["position"]:]}\n'
                    f'B.{doc["manually_fixed_incorrect_sentence"][doc["position"]:]}\nResposta:'
            )
        else:
            return (f'{doc["manually_fixed_correct_sentence"][:doc["position"]]}\n'
                    f'A.{doc["manually_fixed_incorrect_sentence"][doc["position"]:]}\n'
                    f'B.{doc["manually_fixed_correct_sentence"][doc["position"]:]}\nResposta:'
            )

    def doc_to_target(self, doc):
        if doc["question_id"] % 2 == 0:
            return ' A.'
        else:
            return ' B.'

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
        target = self.doc_to_target(doc).strip()
        pred = results[0].strip()
        
        # regex processing. Useful for zero-shot
        match_1 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([AB])\.', pred)
        match_2 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([AB])', pred)
        if len(match_1) > 0:
            pred = match_1[-1] + '.'
        elif len(match_2) > 0:
            pred = match_2[-1] + '.'
        else:
            print(f'Regex failed at processing {pred=}')
            print(f'{target=}, {pred=}')

        return {
            "acc": pred == target
        }
        
