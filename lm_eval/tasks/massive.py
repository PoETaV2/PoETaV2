"""
MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
https://arxiv.org/pdf/2204.08582.pdf

The MASSIVE dataset—Multilingual Amazon Slu resource package (SLURP) for Slot-
filling, Intent classification, and Virtual assistant Evaluation. MASSIVE
contains 1M realistic, parallel, labeled virtual assistant utterances spanning 
51 languages, 18 domains, 60 intents, and 55 slots. MASSIVE was created by 
tasking professional translators to localize the English-only SLURP dataset
into 50 typologically diverse languages from 29 genera. 

In this task we are using only the Portuguese subset, and evaluating only
the domains/scenarios.

Homepage: "https://eval.ai/web/challenges/challenge-page/1697/overview"
"""
import collections
import json
import numpy as np
import os
import requests
from sklearn.metrics import f1_score
import tarfile
from lm_eval.base import rf, PromptSelectionTask
from ..metrics import mean


_CITATION = """
@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages}, 
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_manual_examples = [
    {'id': '1', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'alarm', 'intent': 'alarm_set', 'utt': 'acorda-me às nove da manhã na sexta-feira', 'annot_utt': 'acorda-me às [time : nove da manhã] na [date : sexta-feira]', 'worker_id': '14', 'slot_method': [{'slot': 'time', 'method': 'localization'}, {'slot': 'date', 'method': 'translation'}], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '4878', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'alarm', 'intent': 'alarm_set', 'utt': 'troca o meu alarme das nove da noite por um às cinco da manhã', 'annot_utt': 'troca o meu alarme das [time : nove da noite] por um às [time : cinco da manhã]', 'worker_id': '8', 'slot_method': [{'slot': 'time', 'method': 'localization'}], 'judgments': [{'worker_id': '6', 'intent_score': 2, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '9', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'iot', 'intent': 'iot_hue_lightchange', 'utt': 'tornar a iluminação um pouco mais calorosa aqui', 'annot_utt': 'tornar a iluminação um pouco mais [color_type : calorosa] aqui', 'worker_id': '14', 'slot_method': [{'slot': 'color_type', 'method': 'translation'}], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '18', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'iot', 'intent': 'iot_hue_lightoff', 'utt': 'apagar as luzes no quarto', 'annot_utt': 'apagar as luzes no [house_place : quarto]', 'worker_id': '14', 'slot_method': [{'slot': 'house_place', 'method': 'translation'}], 'judgments': [{'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '33', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'calendar', 'intent': 'calendar_query', 'utt': 'verifique quando o show inicia', 'annot_utt': 'verifique quando o show inicia', 'worker_id': '7', 'slot_method': [], 'judgments': [{'worker_id': '12', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '15', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '9115', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'calendar', 'intent': 'calendar_query', 'utt': 'a que horas é a minha reunião de projecto', 'annot_utt': 'a que horas é a minha [event_name : reunião de projecto]', 'worker_id': '6', 'slot_method': [{'slot': 'event_name', 'method': 'translation'}], 'judgments': [{'worker_id': '3', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '5253', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'play', 'intent': 'play_music', 'utt': 'toca música da minha playlist por favor', 'annot_utt': 'toca música da minha playlist por favor', 'worker_id': '8', 'slot_method': [], 'judgments': [{'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target|english'}]},
    {'id': '9319', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'play', 'intent': 'play_radio', 'utt': 'liga na rádio comercial por favor', 'annot_utt': 'liga na [radio_name : rádio comercial] por favor', 'worker_id': '8', 'slot_method': [{'slot': 'radio_name', 'method': 'localization'}], 'judgments': [{'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '13532', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'general', 'intent': 'general_quirky', 'utt': 'envia flores à minha amiga', 'annot_utt': 'envia flores à minha amiga', 'worker_id': '17', 'slot_method': [], 'judgments': [{'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 0, 'slots_score': 2, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '526', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'general', 'intent': 'general_joke', 'utt': 'conta me anedota aleatória', 'annot_utt': 'conta me anedota [joke_type : aleatória]', 'worker_id': '16', 'slot_method': [{'slot': 'joke_type', 'method': 'translation'}], 'judgments': [{'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 1, 'language_identification': 'target'}]},

    {'id': '4215', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'datetime', 'intent': 'datetime_convert', 'utt': 'preciso do fuso horário de londres inglaterra em vez do fuso horário central', 'annot_utt': 'preciso do fuso horário de [time_zone : londres inglaterra] em vez do fuso horário [time_zone : central]', 'worker_id': '14', 'slot_method': [{'slot': 'time_zone', 'method': 'translation'}], 'judgments': [{'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '4858', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'datetime', 'intent': 'datetime_query', 'utt': 'que dia da semana foi dia quatro de fevereiro de mil novecentos e setenta e dois', 'annot_utt': 'que dia da semana foi [date : dia quatro de fevereiro de mil novecentos e setenta e dois]', 'worker_id': '12', 'slot_method': [{'slot': 'date', 'method': 'translation'}], 'judgments': [{'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '12115', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'recommendation', 'intent': 'recommendation_locations', 'utt': 'encontra sapatarias próximas de minha casa', 'annot_utt': 'encontra [business_type : sapatarias] próximas de minha [place_name : casa]', 'worker_id': '16', 'slot_method': [{'slot': 'business_type', 'method': 'localization'}, {'slot': 'place_name', 'method': 'translation'}], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '3', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '11804', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'recommendation', 'intent': 'recommendation_events', 'utt': 'podes encontrar me uma feira de rua no bairro', 'annot_utt': 'podes encontrar me uma [event_name : feira de rua] no [place_name : bairro]', 'worker_id': '16', 'slot_method': [{'slot': 'event_name', 'method': 'translation'}, {'slot': 'place_name', 'method': 'translation'}], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '3', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '2697', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'takeaway', 'intent': 'takeaway_order', 'utt': 'pede duas pizzas de queijo médias e uma pizza pepperoni média da telepizza', 'annot_utt': 'pede duas pizzas [food_type : de queijo] médias e uma [food_type : pizza pepperoni] média da [business_name : telepizza]', 'worker_id': '8', 'slot_method': [{'slot': 'food_type', 'method': 'translation'}, {'slot': 'business_name', 'method': 'localization'}], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target|other'}]},
    {'id': '3290', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'takeaway', 'intent': 'takeaway_order', 'utt': 'agendar uma entrega de comida para levar em uma lanchonete próxima', 'annot_utt': 'agendar uma [order_type : entrega] de comida [order_type : para levar] em uma [meal_type : lanchonete] próxima', 'worker_id': '2', 'slot_method': [{'slot': 'order_type', 'method': 'translation'}, {'slot': 'meal_type', 'method': 'translation'}], 'judgments': [{'worker_id': '8', 'intent_score': 0, 'slots_score': 0, 'grammar_score': 2, 'spelling_score': 2, 'language_identification': 'target|other'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 2, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '3810', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'audio', 'intent': 'audio_volume_up', 'utt': 'aumentar o volume do altofalante', 'annot_utt': 'aumentar o volume do altofalante', 'worker_id': '11', 'slot_method': [], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '4397', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'audio', 'intent': 'audio_volume_down', 'utt': 'podes falar mais baixo por favor', 'annot_utt': 'podes falar mais baixo por favor', 'worker_id': '12', 'slot_method': [], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '13', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '12876', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'transport', 'intent': 'transport_taxi', 'utt': 'ligue para táxis de lisboa e reserve um táxi para três horas em ponto da manhã', 'annot_utt': 'ligue para [business_name : táxis de lisboa] e reserve um [transport_type : táxi] para [time : três] horas em ponto da [timeofday : manhã]', 'worker_id': '0', 'slot_method': [{'slot': 'business_name', 'method': 'localization'}, {'slot': 'transport_type', 'method': 'unchanged'}, {'slot': 'time', 'method': 'translation'}, {'slot': 'timeofday', 'method': 'translation'}], 'judgments': [{'worker_id': '11', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '12246', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'transport', 'intent': 'transport_query', 'utt': 'existem trem saindo de porto para lisboa ao meio-dia', 'annot_utt': 'existem [transport_type : trem] saindo de [place_name : porto] para [place_name : lisboa] ao [timeofday : meio-dia]', 'worker_id': '11', 'slot_method': [{'slot': 'transport_type', 'method': 'translation'}, {'slot': 'place_name', 'method': 'localization'}, {'slot': 'timeofday', 'method': 'translation'}], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '11068', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'lists', 'intent': 'lists_createoradd', 'utt': 'preciso de acrescentar laranjas à minha lista de mercearia', 'annot_utt': 'preciso de acrescentar laranjas à minha lista de [list_name : mercearia]', 'worker_id': '2', 'slot_method': [{'slot': 'list_name', 'method': 'translation'}], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '8', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 2, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '10680', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'lists', 'intent': 'lists_query', 'utt': 'diz-me qual é a minha lista de afazeres hoje', 'annot_utt': 'diz-me qual é a minha lista de [list_name : afazeres] [date : hoje]', 'worker_id': '2', 'slot_method': [{'slot': 'list_name', 'method': 'translation'}, {'slot': 'date', 'method': 'translation'}], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '3025', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'weather', 'intent': 'weather_query', 'utt': 'tempo no porto', 'annot_utt': 'tempo no [place_name : porto]', 'worker_id': '12', 'slot_method': [{'slot': 'place_name', 'method': 'localization'}], 'judgments': [{'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '3', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '3548', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'weather', 'intent': 'weather_query', 'utt': 'quando vai nevar novamente', 'annot_utt': 'quando vai [weather_descriptor : nevar] novamente', 'worker_id': '12', 'slot_method': [{'slot': 'weather_descriptor', 'method': 'translation'}], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '9957', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'cooking', 'intent': 'cooking_recipe', 'utt': 'quanto tempo deve cozinhar um frango inteiro num forno de trezentos e setenta e cinco graus', 'annot_utt': 'quanto tempo deve cozinhar um [food_type : frango inteiro] num forno de trezentos e setenta e cinco graus', 'worker_id': '15', 'slot_method': [{'slot': 'food_type', 'method': 'translation'}], 'judgments': [{'worker_id': '18', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '10084', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'cooking', 'intent': 'cooking_recipe', 'utt': 'o que é uma alternativa à farinha', 'annot_utt': 'o que é uma [ingredient : alternativa à farinha]', 'worker_id': '2', 'slot_method': [{'slot': 'ingredient', 'method': 'translation'}], 'judgments': [{'worker_id': '17', 'intent_score': 0, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 2, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},

    {'id': '15574', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'social', 'intent': 'social_post', 'utt': 'abra twitter das estradas de portugal', 'annot_utt': 'abra [media_type : twitter] das [business_name : estradas de portugal]', 'worker_id': '11', 'slot_method': [{'slot': 'media_type', 'method': 'localization'}, {'slot': 'business_name', 'method': 'localization'}], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target|english'}, {'worker_id': '2', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target|english'}, {'worker_id': '18', 'intent_score': 0, 'slots_score': 0, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '15398', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'social', 'intent': 'social_post', 'utt': "envie uma reclamação para mount's sobre problemas de velocidade da internet", 'annot_utt': "envie uma reclamação para [business_name : mount's] sobre problemas de velocidade da internet", 'worker_id': '11', 'slot_method': [{'slot': 'business_name', 'method': 'localization'}], 'judgments': [{'worker_id': '2', 'intent_score': 1, 'slots_score': 0, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target|english'}, {'worker_id': '18', 'intent_score': 0, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target|english'}]},

    {'id': '16670', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'email', 'intent': 'email_query', 'utt': 'quantos emails não lidos tenho neste momento', 'annot_utt': 'quantos emails não lidos tenho neste momento', 'worker_id': '14', 'slot_method': [], 'judgments': [{'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target|english'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 2, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '16618', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'email', 'intent': 'email_sendemail', 'utt': 'redigir um email para carina sobre o trabalho mais tarde', 'annot_utt': 'redigir um email para [person : carina] sobre o trabalho mais tarde', 'worker_id': '6', 'slot_method': [{'slot': 'person', 'method': 'unchanged'}], 'judgments': [{'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target|english'}]},

    {'id': '14881', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'qa', 'intent': 'qa_currency', 'utt': 'qual a proporção da rúpia indiana para o dólar americano', 'annot_utt': 'qual a proporção da [currency_name : rúpia indiana] para o [currency_name : dólar americano]', 'worker_id': '7', 'slot_method': [{'slot': 'currency_name', 'method': 'translation'}], 'judgments': [{'worker_id': '15', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '3', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 3, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}]},
    {'id': '13070', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'qa', 'intent': 'qa_definition', 'utt': 'siri quais são as definições de laranja', 'annot_utt': 'siri quais são as definições de [definition_word : laranja]', 'worker_id': '2', 'slot_method': [{'slot': 'definition_word', 'method': 'translation'}], 'judgments': [{'worker_id': '2', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '11', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target|english'}]},

    {'id': '4860', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'news', 'intent': 'news_query', 'utt': 'diz-me quais são as noticias mais recentes', 'annot_utt': 'diz-me quais são as noticias mais recentes', 'worker_id': '14', 'slot_method': [], 'judgments': [{'worker_id': '8', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 2, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}]},
    {'id': '2640', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'news', 'intent': 'news_query', 'utt': 'fale-me sobre panama papers', 'annot_utt': 'fale-me sobre [news_topic : panama papers]', 'worker_id': '7', 'slot_method': [{'slot': 'news_topic', 'method': 'localization'}], 'judgments': [{'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target|english'}, {'worker_id': '6', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target'}, {'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 1, 'language_identification': 'target|english'}]},

    {'id': '78', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'music', 'intent': 'music_likeness', 'utt': 'a música que estás a tocar é incrível', 'annot_utt': 'a música que estás a tocar é incrível', 'worker_id': '16', 'slot_method': [], 'judgments': [{'worker_id': '0', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
    {'id': '3939', 'locale': 'pt-PT', 'partition': 'train', 'scenario': 'music', 'intent': 'music_likeness', 'utt': 'podes guardar esta música nos meus favoritos', 'annot_utt': 'podes guardar esta música nos meus favoritos', 'worker_id': '16', 'slot_method': [], 'judgments': [{'worker_id': '12', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '14', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}, {'worker_id': '16', 'intent_score': 1, 'slots_score': 1, 'grammar_score': 4, 'spelling_score': 2, 'language_identification': 'target'}]},
]

class MASSIVE(PromptSelectionTask):
    VERSION = 0
    DATASET_PATH = "data/MASSIVE"
    DATASET_NAME = None
    
    KEYS_TO_INDEX = ['utt']
    KEY_TO_BALANCE = 'scenario'
    NUM_CLASSES = 18
    SEARCHER_K = 600

    manual_examples = _manual_examples

    scenarios = {
        'general':          'geral',
        'datetime':         'data/hora',
        'recommendation':   'recomendação',
        'takeaway':         'entrega',
        'audio':            'áudio',
        'calendar':         'calendário',
        'transport':        'transporte',
        'lists':            'listas',
        'weather':          'clima',
        'iot':              'IoT',
        'play':             'tocar/jogar',
        'cooking':          'culinária',
        'social':           'social',
        'email':            'email',
        'qa':               'perguntas e respostas',
        'alarm':            'alarme',
        'news':             'notícias',
        'music':            'música'
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = collections.defaultdict(list)
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH, exist_ok=True)
            URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
            response = requests.get(URL, stream=True)
            file = tarfile.open(fileobj=response.raw, mode="r|gz")
            file.extractall(path=self.DATASET_PATH)

        with open(os.path.join(self.DATASET_PATH, '1.0', 'data', 'pt-PT.jsonl')) as f:
            for line in f:
                doc = json.loads(line)
                self.dataset[doc['partition']].append(doc)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["dev"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"Comando: {doc['utt']}\nCategoria:"

    def doc_to_target(self, doc):
        return " " + self.scenarios[doc['scenario']]

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
        for scenario in self.scenarios.values():
            ll, _ = rf.loglikelihood(ctx, " " + scenario)
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
        gold = list(self.scenarios).index(doc['scenario'])
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

class MASSIVE_GREEDY(MASSIVE):
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
        gold = self.doc_to_target(doc).strip()
        pred = results[0].strip()

        return {
            "acc": (pred == gold) * 100.0,
            "f1-macro": (pred, gold),
            "f1-weighted": (pred, gold),
        }
