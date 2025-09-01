"""
MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering
https://arxiv.org/abs/2007.15207

MKQA is an open-domain question answering evaluation set comprising 10k 
question-answer pairs aligned across 26 typologically diverse languages (260k 
question-answer pairs in total). The goal of this dataset is to provide a 
challenging benchmark for question answering quality across a wide set of 
languages.

In this task we are using only the Portuguese queries. Additionally, we 
adopt closed-book question-answering (not using NQ contexts), and ignore the 
examples of `unanswerable` and `long_answers` types. Finally, we evaluate the 
model using in-context learning with leave-one-out validation protocol.

"""
import collections
import copy
from gzip import GzipFile
import re
import os
import json
from math import exp
from lm_eval.base import rf, PromptSelectionTask
from functools import partial
from lm_eval.metrics import yesno
from lm_eval.custom_eval.mkqa import mkqa_eval
from urllib.request import urlretrieve


_CITATION = """
@misc{mkqa,
    title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
    author = {Shayne Longpre and Yi Lu and Joachim Daiber},
    year = {2020},
    URL = {https://arxiv.org/pdf/2007.15207.pdf}
}
"""

def _mkqa_agg(key, items):

    annotations = mkqa_eval.read_annotations('data/mkqa/mkqa.jsonl.gz')['pt']
    predictions = mkqa_eval.process_predictions(items)
    metrics = mkqa_eval.evaluate(
        annotations, predictions, language='pt', verbose=False, print_metrics=False
    )

    return metrics[key]

_manual_examples = [
    # number_with_unit
    {'example_id': '3051930912491995402', 'query': 'quanto tempo levou para as torres gêmeas serem construídas', 'type': 'number_with_unit', 'answers': ['11 ano', '11.0 ano']},
    {'example_id': '-1613565801972062441', 'query': 'qual é o diâmetro da Terra no equador', 'type': 'number_with_unit', 'answers': ['12742 quilómetros', '12742.0 quilómetros', '7917.5 milha']},
    {'example_id': '7353261961888670371', 'query': 'qual é o recorde de temperatura mais quente na terra', 'type': 'number_with_unit', 'answers': ['134.1 Escala Farenheit']},
    {'example_id': '2074109525231919974', 'query': 'quantos minutos tem um round no boxe', 'type': 'number_with_unit', 'answers': ['3 minuto', '3.0 minuto']},
    {'example_id': '3822674477738969189', 'query': 'com quantos anos a rainha elizabeth se tornou rainha', 'type': 'number_with_unit', 'answers': ['27 ano', '27.0 ano']},
    {'example_id': '-3845254214087419206', 'query': 'quanto tempo é o prazo para um senador dos EUA', 'type': 'number_with_unit', 'answers': ['6 ano', '6.0 ano']},
    {'example_id': '182673980570613877', 'query': 'qual é a altura atual do monte everest', 'type': 'number_with_unit', 'answers': ['8848 metros', '8848.0 metros']},
    {'example_id': '4830900147895398824', 'query': 'qual a altura da torre do terror de hollywood', 'type': 'number_with_unit', 'answers': ['199 pé', '199.0 pé', '61 metros', '61.0 metros']},
    {'example_id': '7728506664410615612', 'query': 'quantos litros em um barril de cerveja', 'type': 'number_with_unit', 'answers': ['31.5 Galão imperial']},
    {'example_id': '-6755191382094968028', 'query': 'qual a distância da inglaterra a dubai', 'type': 'number_with_unit', 'answers': ['4511 milha', '4511.0 milha']},
    {'example_id': '4997833347066913620', 'query': 'quanta terra o México perdeu na Guerra Mexicano-Americana', 'type': 'number_with_unit', 'answers': ['525000 milha quadrada', '525000.0 milha quadrada']},
    # # long_answer
    # {'example_id': '-7304115666479779940', 'query': 'de onde vem a expressão ir pro brejo', 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '2018272005970792683', 'query': "de onde veio o termo 'let them eat the cake'", 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '-512527307772390265', 'query': 'por que chamamos new york de big apple', 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '-2077080565193970597', 'query': 'onde o rio de ohio começa e para', 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '9002735418234562192', 'query': 'qual é a diferença entre umidade e ponto de condensação da água', 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '-5936160720016042676', 'query': 'o que o chefe de justiça faz', 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '3854590860731408453', 'query': 'qual foi o resultado da guerra franco-indígena', 'type': 'long_answer', 'answers': ['N/A']},
    # {'example_id': '4791591886298092096', 'query': 'qual é a ordem de filmes dos vingadores marvel', 'type': 'long_answer', 'answers': ['N/A']},
    # entity
    {'example_id': '3215481144081056949', 'query': 'quem escreveu o livro das lamentações na bíblia', 'type': 'entity', 'answers': ['Jeremias', 'Profeta Jeremias']},
    {'example_id': '3054373585607347258', 'query': 'quem interpretou o pantera negra no filme pantera negra', 'type': 'entity', 'answers': ['Chadwick Aaron Boseman', 'Chadwick Boseman']},
    {'example_id': '-2162714330037272999', 'query': "quem cantava what is love baby don't hurt me", 'type': 'entity', 'answers': ['Haddaway']},
    {'example_id': '-7038357099440704798', 'query': 'Qual é o ingrediente ativo em todos os higienizadores de mãos aprovados pela FDA', 'type': 'entity', 'answers': ['C2H6O', 'etanol', 'Álcool etílico']},
    {'example_id': '301001421963967136', 'query': 'quem cantou a música take me to church', 'type': 'entity', 'answers': ['Hozier']},
    {'example_id': '8259032885448966485', 'query': 'quem era o homem mais velho que já viveu', 'type': 'entity', 'answers': ['Jeanne Calment', 'Jiroemon Kimura']},
    {'example_id': '5599669105153346278', 'query': 'quem é o ator que faz o john wick', 'type': 'entity', 'answers': ['Keanu Charles Reeves', 'Keanu Reeves']},
    {'example_id': '2972778292233187574', 'query': 'onde é filmado o programa de tv stranger things', 'type': 'entity', 'answers': ['Atlanta', 'Geórgia']},
    {'example_id': '4174294807029812925', 'query': 'qual é a capital do estado do novo méxico', 'type': 'entity', 'answers': ['Santa Fe', 'Santa Fé']},
    {'example_id': '-4541286593031074150', 'query': 'quem foi a primeira pessoa que andou na lua', 'type': 'entity', 'answers': ['Neil Armstrong']},
    {'example_id': '2389510054222243562', 'query': 'quem interpreta gandalf no senhor dos anéis', 'type': 'entity', 'answers': ['Ian McKellen', 'Sir Ian McKellen']},
    # number
    {'example_id': '5977557104954330791', 'query': 'quantos jogos da NBA cada time joga', 'type': 'number', 'answers': ['82', '82.0']},
    {'example_id': '-4566536088537382616', 'query': 'quantos dias santos de obrigação existem na Igreja Católica', 'type': 'number', 'answers': ['6', '6.0']},
    {'example_id': '-3781524301475508699', 'query': 'quantas tropas americanas morreram na guerra do vietnã', 'type': 'number', 'answers': ['57939', '57939.0', '58220', '58220.0']},
    {'example_id': '4269706947288164231', 'query': 'Quantos planetas estão em nosso sistema solar?', 'type': 'number', 'answers': ['8', '8.0']},
    {'example_id': '5627813778241033847', 'query': 'quantos votos cada estado tem sob os artigos da confederação', 'type': 'number', 'answers': ['1', '1.0']},
    {'example_id': '4080291615488399486', 'query': 'quantos ossos um ser humano tem em seu corpo', 'type': 'number', 'answers': ['206', '206.0']},
    {'example_id': '-3258639610940537785', 'query': 'quantas temporadas de game of thrones houve', 'type': 'number', 'answers': ['8', '8.0']},
    {'example_id': '8799194996341417721', 'query': 'quantos filmes Debi & Loide  tem', 'type': 'number', 'answers': ['3', '3.0']},
    {'example_id': '4830453860214251098', 'query': 'quantas voltas há no indy 500', 'type': 'number', 'answers': ['200', '200.0']},
    {'example_id': '-947675951118523690', 'query': 'quantos tipos de queijo existem no mundo', 'type': 'number', 'answers': ['7', '7.0']},
    {'example_id': '6265990105062828626', 'query': 'quantos gols ronaldo marcou na premier league', 'type': 'number', 'answers': ['84', '84.0']},
    # date
    {'example_id': '1617928488861000814', 'query': 'quando o samsung galaxy s6 edge foi lançado', 'type': 'date', 'answers': ['04 10 2015', '04-10-2015', '04/10/2015', '10 04 2015', '10 abril 2015', '10-04-2015', '10/04/2015', '2015 04 10', '2015 abril 10', '2015 april, 10', '2015-04-10']},
    {'example_id': '2993900242277777418', 'query': 'quando foi construido a estátua da liberdade?', 'type': 'date', 'answers': ['1875 09', '1875 setembro', '1875-09', 'september, 1875', 'setembro 1875']},
    {'example_id': '-8272431585098222113', 'query': 'quando foi lançado o filme Pantera Negra', 'type': 'date', 'answers': ['02 16 2018', '02-16-2018', '02/16/2018', '16 02 2018', '16 fevereiro 2018', '16-02-2018', '16/02/2018', '2018 02 16', '2018 february, 16', '2018 fevereiro 16', '2018-02-16']},
    {'example_id': '4810976600081985638', 'query': 'quando o filme original do godzilla foi lançado', 'type': 'date', 'answers': ['10 27 1954', '10-27-1954', '10/27/1954', '1954 10 27', '1954 october, 27', '1954 outubro 27', '1954-10-27', '27 10 1954', '27 outubro 1954', '27-10-1954', '27/10/1954']},
    {'example_id': '-3721135145130183624', 'query': 'quando a Pérsia mudou seu nome para Irã', 'type': 'date', 'answers': ['1935']},
    {'example_id': '744447684240531341', 'query': 'quando saiu o primeiro telefone touchscreen', 'type': 'date', 'answers': ['1992']},
    {'example_id': '4342171995664797297', 'query': 'quando foi lançada a tv em preto e branco', 'type': 'date', 'answers': ['1936']},
    {'example_id': '571452456968526218', 'query': 'quando foi a última vez que o paquistão venceu a copa do mundo', 'type': 'date', 'answers': ['1992']},
    {'example_id': '-6933941057494901788', 'query': 'quando o world trade center foi construído', 'type': 'date', 'answers': ['1973']},
    {'example_id': '4136105964680190276', 'query': 'quando foi lançado star wars uma nova esperança', 'type': 'date', 'answers': ['05 25 1977', '05-25-1977', '05/25/1977', '1977 05 25', '1977 maio 25', '1977 may, 25', '1977-05-25', '25 05 1977', '25 maio 1977', '25-05-1977', '25/05/1977']},
    {'example_id': '-7370774985713456088', 'query': 'Quando foi assinada a constituição americana?', 'type': 'date', 'answers': ['09 17 1787', '09-17-1787', '09/17/1787', '17 09 1787', '17 setembro 1787', '17-09-1787', '17/09/1787', '1787 09 17', '1787 september, 17', '1787 setembro 17', '1787-09-17']},
    # # unanswerable
    # {'example_id': '-8048076596333046562', 'query': 'quem foi a primeira pessoa a fazer um slime', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '6016187989170734632', 'query': 'como chego à rodovia 85 daqui', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '3699573018209589701', 'query': 'qual o nome da tempestade no atlântico', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '-8822259047965110671', 'query': 'leve-me ao aeroporto charlotte north carolina', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '4683798436554413780', 'query': 'Quão longe é daqui a Oklahoma City?', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '5224214262209254277', 'query': 'me mostre uma foto do lanterna verde', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '-4941663564532394007', 'query': 'mostre-me um mapa dos países do Oriente Médio', 'type': 'unanswerable', 'answers': ['N/A']},
    # {'example_id': '-803285612943718200', 'query': 'quem jogou o primeiro jogo de futebol universitário esse ano', 'type': 'unanswerable', 'answers': ['N/A']},
    # short_phrase
    {'example_id': '-1900131123284252568', 'query': 'quem é o irmão em everybody loves raymond', 'type': 'short_phrase', 'answers': ['Robert Barone']},
    {'example_id': '-6930371014100116329', 'query': 'qual o nome do macaco de aladdin', 'type': 'short_phrase', 'answers': ['Abu']},
    {'example_id': '8236593213458204693', 'query': 'Como você chama uma pessoa que prevê o futuro?', 'type': 'short_phrase', 'answers': ['Médium', 'Profeta', 'vidente', 'Áugure']},
    {'example_id': '-5312195282561428201', 'query': 'qual o nome do tubarão em procurando nemo', 'type': 'short_phrase', 'answers': ['Bruce']},
    {'example_id': '5488771502904120932', 'query': 'qual é o nome do príncipe em A bela e a fera?', 'type': 'short_phrase', 'answers': ['Adam']},
    {'example_id': '-5388681249122308164', 'query': 'quando foi feita a música cotton eyed joe', 'type': 'short_phrase', 'answers': ['Antes de 1861']},
    {'example_id': '6134810927991933545', 'query': 'qual era o nome da van de scooby doo', 'type': 'short_phrase', 'answers': ['A Máquina de Mistério']},
    {'example_id': '-5441366966986750352', 'query': 'onde diz na bíblia sobre os dez mandamentos', 'type': 'short_phrase', 'answers': ['Deuteronômio 5:4-21', 'Êxodo 20:1-17', 'Êxodo 31:18']},
    {'example_id': '3654252407672083306', 'query': 'que música Bob Esponja cantava no Bubble Bowl', 'type': 'short_phrase', 'answers': ['sweet victory']},
    {'example_id': '3164205542785721140', 'query': 'nós sempre acreditamos no que queremos acreditar em latim', 'type': 'short_phrase', 'answers': ['Credimus quod credere volumus']},
    {'example_id': '4807982044372192071', 'query': 'como é o número 8 em algarismos romanos', 'type': 'short_phrase', 'answers': ['VIII']},
    # binary
    {'example_id': '6734439646325158985', 'query': 'já houve um furacão chamado kathy?', 'type': 'binary', 'answers': ['yes']},
    {'example_id': '2639735427364036124', 'query': 'o havaí faz parte do continente norte americano', 'type': 'binary', 'answers': ['yes']},
    {'example_id': '7676010155159524765', 'query': 'tem um livro de judas na bíblia', 'type': 'binary', 'answers': ['yes']},
    {'example_id': '-4478079202372445991', 'query': 'é legal casar com sua irmã no alabama', 'type': 'binary', 'answers': ['yes']},
    {'example_id': '-129067025216529276', 'query': 'o monte everest é a montanha mais alta do mundo', 'type': 'binary', 'answers': ['yes']},
    {'example_id': '6771093663807806032', 'query': 'é obrigatório ir à escola no canadá', 'type': 'binary', 'answers': ['yes']},
    {'example_id': '-8619396784418257389', 'query': 'costa rica é um território dos estados unidos', 'type': 'binary', 'answers': ['no']},
    {'example_id': '-204511291662634757', 'query': 'pode a rainha no xadrez se mover como o cavalo', 'type': 'binary', 'answers': ['no']},
    {'example_id': '5999602363334169462', 'query': 'o grand canyon é o maior desfiladeiro do mundo', 'type': 'binary', 'answers': ['no']},
    {'example_id': '6416172900659432472', 'query': 'Os cidadãos australianos precisam de visto para viajar para a Itália?', 'type': 'binary', 'answers': ['no']},
    {'example_id': '6521550237225575378', 'query': 'Logan é a última parte do xmen?', 'type': 'binary', 'answers': ['no']},
]

class MKQA(PromptSelectionTask):
    VERSION = 1
    DATASET_PATH = "data/mkqa"
    
    KEYS_TO_INDEX = ['query']
    KEY_TO_BALANCE = None
    NUM_CLASSES = 8
    SEARCHER_K = 600

    manual_examples = _manual_examples

    exclude_unanswerables_and_longanswers = True

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """We could download the dataset directly from the datasets library.
        However, it seems that the answer types are not consistent with the 
        original dataset (using integer instead of string), and the original 
        jsonl file is needed to use the mkqa evaluation script.
        """
        if os.path.exists(self.DATASET_PATH):
            print(f"Reusing dataset mkqa ({self.DATASET_PATH})")
        else:
            os.makedirs(self.DATASET_PATH, exist_ok=True)
            urlretrieve('https://github.com/apple/ml-mkqa/raw/651b8cc85c407270b024157aff06ee6ab8c4fc6d/dataset/mkqa.jsonl.gz',
                        os.path.join(self.DATASET_PATH, 'mkqa.jsonl.gz'))

        self.dataset = collections.defaultdict(list)
        gzipped_input_file = open(os.path.join(self.DATASET_PATH, 'mkqa.jsonl.gz'), "rb")
        with GzipFile(fileobj=gzipped_input_file) as input_file:
            for line in input_file:
                doc = json.loads(line)

                # Simplifying the data structure.
                valid_answers, answer_types = [], []
                for answer in doc["answers"]['pt']:
                    # Binary (Yes/No) answer text is always "yes" / "no"
                    # If answer['text'] is None then it `"N/A"` represents No Answer
                    valid_answers.append(answer["text"] or "N/A")
                    valid_answers.extend(answer.get("aliases", []))
                    answer_types.append(answer["type"])

                # Sorting the answers to make the prompts deterministic. The 
                # first answer will be used in doc_to_target.
                valid_answers = list(set(valid_answers))
                valid_answers.sort()

                example = {
                    'example_id': str(doc['example_id']),
                    'query': doc['queries']['pt'],
                    'type': answer_types[0],
                    'answers': valid_answers,
                }

                if self.exclude_unanswerables_and_longanswers and answer_types[0] in ['long_answer', 'unanswerable']:
                    continue

                self.dataset['train'].append(example)

    def create_collection_to_index(self):
        """ Creates a JSON collection to index. Overwrite this funtion to keep
        more arguments.
        """
        json_data = []
        for i, doc in enumerate(self.dataset['train']):
            assert all(arg in doc for arg in self.KEYS_TO_INDEX), (
                "The keys to be indexed must be present in all the documents")
            assert self.KEY_TO_BALANCE in doc or self.KEY_TO_BALANCE is None, (
                "The KEY_TO_BALANCE must be present in all the documents"
            )

            data = {
                'id': i,
                'contents': '. '.join(doc[key] for key in self.KEYS_TO_INDEX),
                'example_id': doc['example_id'],  # used to filter out the current document from prompt candidates
            }
            if self.KEY_TO_BALANCE:
                data[self.KEY_TO_BALANCE] = doc[self.KEY_TO_BALANCE]

            json_data.append(data)

        with open(self.documents_to_index, 'w') as f:
            json.dump(json_data, f)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset['train']

    def test_docs(self):
        return self.dataset["train"]

    def doc_to_text(self, doc):
        query = doc['query']
        if not query.endswith('?'):
            query += '?'
        return 'Pergunta: ' + query + '\n' + 'Resposta:'

    def doc_to_target(self, doc):
        answer = doc['answers'][0]
        if answer == 'yes':
            return " sim"
        if answer == 'no':
            return " não"
        return " " + answer

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
        prediction = rf.greedy_until(ctx, ['\n'])

        ll_yes, _ = rf.loglikelihood(ctx, " sim")
        ll_no, _ = rf.loglikelihood(ctx, " não")

        is_unanswerable, _ = rf.loglikelihood(ctx, " N/A")
        return prediction, ll_yes, ll_no, is_unanswerable
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        prediction, logprob_yes, logprob_no, logprob_unanswerable = results

        binary_answer = yesno(logprob_yes > logprob_no)
        no_answer_prob = exp(logprob_unanswerable)

        answer_type = doc["type"]
        if answer_type == 'binary':
            prediction = None
        else:
            binary_answer = None

        prediction = {
            "example_id": doc['example_id'], "prediction": prediction,
            "binary_answer": binary_answer, "no_answer_prob": no_answer_prob
        }

        return { 
            'best_em': prediction,
            'best_f1': prediction,
            'best_answerable_em': prediction,
            'best_answerable_f1': prediction,
            'best_unanswerable_em': prediction,
            'best_f1_threshold': prediction,
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return { 
            'best_em': partial(_mkqa_agg, 'best_em'), 
            'best_f1': partial(_mkqa_agg, 'best_f1'),
            'best_answerable_em': partial(_mkqa_agg, 'best_answerable_em'),
            'best_answerable_f1': partial(_mkqa_agg, 'best_answerable_f1'),
            'best_unanswerable_em': partial(_mkqa_agg, 'best_unanswerable_em'),
            'best_f1_threshold': partial(_mkqa_agg, 'best_f1_threshold'),
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return { 
            'best_em': True,
            'best_f1': True,
            'best_answerable_em': True,
            'best_answerable_f1': True,
            'best_unanswerable_em': True,
            'best_f1_threshold': True,
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

            hits = self.searcher.search(doc['query'], k=self.SEARCHER_K)
            indices = []

            for hit in hits:
                hit = json.loads(hit.raw)

                # Ignore the current document
                if hit['example_id'] == doc['example_id']:
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

    #####################################################################################
    # Below is the version of fewshot_examples that works for "KEY_TO_BALANCE = 'type'"
    #####################################################################################
    # def fewshot_examples(self, k, rnd, prompt_mode, doc):
    #     if self._training_docs is None:
    #         self._training_docs = list(self.training_docs())

    #     if prompt_mode == 'dynamic-random':
    #         # Ignore the current document
    #         _training_docs = copy.copy(self.training_docs())
    #         _training_docs.remove(doc)
    #         return rnd.sample(_training_docs, k)

    #     elif prompt_mode == 'fixed':
    #         # We consider the first k // NUM_CLASSES of each class. If doc is among
    #         # them, we ignore and the next document of the same class of doc. 
    #         indices = [id for lst in self.indices_per_class.values() for id in lst[:k // self.NUM_CLASSES]]
    #         # Ignore the current document
    #         examples = [self._training_docs[id] for id in indices if self._training_docs[id] != doc]
    #         if len(examples) <= k:
    #             next_indice = self.indices_per_class[doc[self.KEY_TO_BALANCE]][k // self.NUM_CLASSES]
    #             examples.append(self._training_docs[next_indice])
    #         return rnd.sample(examples, k)

    #     elif prompt_mode == 'manual':
    #         # Check if each class has enough manual examples. 
    #         for cls in self.manual_indices_per_class:
    #             assert k // self.NUM_CLASSES <= len(self.manual_indices_per_class[cls]), (
    #                 f'The number of manual_examples for class {cls} is not enough to satisfy '
    #                 f'num_fewshot={k}. Please, include more examples.')
    #         # We consider the first k // NUM_CLASSES of each class. If doc is among
    #         # them, we ignore and the next document of the same class of doc. 
    #         indices = [id for lst in self.manual_indices_per_class.values() for id in lst[:k // self.NUM_CLASSES]]
    #         # Ignore the current document
    #         examples = [self.manual_examples[id] for id in indices if self._training_docs[id] != doc]
    #         if len(examples) <= k:
    #             next_indice = self.manual_indices_per_class[doc[self.KEY_TO_BALANCE]][k // self.NUM_CLASSES]
    #             examples.append(self.manual_examples[next_indice])
    #         return rnd.sample(examples, k)

    #     elif prompt_mode == 'dynamic-similar':
    #         hits = self.searcher.search(doc['query'], k=self.SEARCHER_K)

    #         indices_per_class = collections.defaultdict(list)
    #         kclass = k // self.NUM_CLASSES  # expected number of examples per class
    #         counter = 0

    #         for hit in hits:
    #             hit = json.loads(hit.raw)
    #             answer_type = hit['type']
    #             id = hit['id']

    #             assert hit['contents'] == self._training_docs[id]['query']

    #             # Ignore the current document
    #             if len(indices_per_class[answer_type]) < kclass and doc['example_id'] != hit['example_id']:
    #                 indices_per_class[answer_type].append(id)
    #                 counter += 1

    #             # stop if we reached the expected number of indices
    #             if counter == kclass * len(self.indices_per_class):
    #                 break

    #         # check if each class has enough similar examples. If not, complete 
    #         # with random examples. Also, move the most relevant examples to the end.
    #         for cls in self.indices_per_class:
    #             _len = len(indices_per_class[cls])
    #             if _len < kclass:
    #                 indices_per_class[cls].extend(
    #                     self.indices_per_class[cls][:kclass-_len])

    #             indices_per_class[cls].reverse()
            
    #         indices = []
    #         classes = list(indices_per_class.keys())
    #         for i in range(kclass):
    #             rnd.shuffle(classes)
    #             for cls in classes:
    #                 indices.append(indices_per_class[cls][i])

    #         return [ self._training_docs[i] for i in indices[:k] ]

    #     else:
    #         print('Please set prompt_mode as "fixed", "dynamic-random", "dynamic-similar", or "manual"')


class MKQA_GREEDY(MKQA):
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
        prediction = rf.greedy_until(ctx, ['\n'])

        return prediction
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        prediction = results[0]
        no_answer_prob = 0
        answer_type = doc["type"]
        
        if answer_type == 'binary':
            positive_answer = re.match(r'\s*(sim).*', prediction, re.IGNORECASE)
            negative_answer = re.match(r'\s*(nao|não).*', prediction, re.IGNORECASE)

            if positive_answer:
                binary_answer = "yes"
            elif negative_answer:
                binary_answer = "no"
            else:
                binary_answer = None

            prediction = None
        else:
            binary_answer = None

        prediction = {
            "example_id": doc['example_id'], "prediction": prediction,
            "binary_answer": binary_answer, "no_answer_prob": no_answer_prob
        }

        return { 
            'best_em': prediction,
            'best_f1': prediction,
            'best_answerable_em': prediction,
            'best_answerable_f1': prediction,
            'best_unanswerable_em': prediction,
            'best_f1_threshold': prediction,
        }
