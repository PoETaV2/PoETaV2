"""
University Entrance Exam as a Guiding Test for Artificial Intelligence
https://www.ime.usp.br/~ddm/project/enem/ENEM-GuidingTest.pdf

The ENEM Challenge consists in designing an autonomous system that matches the 
performance of a human students on the exam. The overall goal is to foster and 
evaluate the development of Artificial Intelligence techniques that have good 
performance on complex cognitive tasks, not particularly designed for AI systems. 
In addition, this challenge aims to promote and give more visiblity to the 
development of NLP tools for Brazilian Portuguese.

Homepage: https://www.ime.usp.br/~ddm/project/enem
"""
import collections
import datasets
from io import BytesIO
import json
import numpy as np
import os
import re
import urllib.request
from urllib.request import urlopen
import xml.etree.ElementTree as ET 
from zipfile import ZipFile

from lm_eval.base import rf, MultipleChoicePromptSelectionTask
from lm_eval.metrics import mean


_CITATION = """
@InProceedings{ ENEM-Challenge,
    author={Silveira, Igor Cataneo and Mau\'a, Denis Deratani},
    booktitle={Proceedings of the 6th Brazilian Conference on Intelligent Systems},
    series={BRACIS},
    title={University Entrance Exam as a Guiding Test for Artificial Intelligence},
    pages={426--431},
    year={2017}
}
"""


PATTERNS_REPLACES = [
    (r'\s*\n+\s*', r' '),  # changing \n to space
    (r'(\s)\1+', r' '),  # changing \n to space
    (r'^\s+', r''),
]


apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)


manual_examples = [
    {'query': 'Cabeçalho: Estima-se que haja atualmente no mundo 40 milhões de pessoas infectadas pelo HIV (o vírus que causa a AIDS), sendo que as taxas de novas infecções continuam crescendo, principalmente na África, Ásia e Rússia. Nesse cenário de pandemia, uma vacina contra o HIV teria imenso impacto, pois salvaria milhões de vidas. Certamente seria um marco na história planetária e também uma esperança para as populações carentes de tratamento antiviral e de acompanhamento médico. TANURI, A.; FERREIRA JUNIOR, O. C. Vacina contra Aids: desafios e esperanças. Ciência Hoje (44) 26, 2009 (adaptado). \nEnunciado: Uma vacina eficiente contra o HIV deveria \nAlternativas:\nA. induzir a imunidade, para proteger o organismo da contaminação viral. \nB. ser capaz de alterar o genoma do organismo portador, induzindo a síntese de enzimas protetoras. \nC. produzir antígenos capazes de se ligarem ao vírus, impedindo que este entre nas células do organismo humano. \nD. ser amplamente aplicada em animais, visto que esses são os principais transmissores do vírus para os seres humanos. \nE. estimular a imunidade, minimizando a transmissão do vírus por gotículas de saliva. \nResposta:', 'choices': ['induzir a imunidade, para proteger o organismo da contaminação viral. ', 'ser capaz de alterar o genoma do organismo portador, induzindo a síntese de enzimas protetoras. ', 'produzir antígenos capazes de se ligarem ao vírus, impedindo que este entre nas células do organismo humano. ', 'ser amplamente aplicada em animais, visto que esses são os principais transmissores do vírus para os seres humanos. ', 'estimular a imunidade, minimizando a transmissão do vírus por gotículas de saliva. '], 'gold': 0, 'id': '2009_03', 'exam': '2009', 'contents': 'Estima-se que haja atualmente no mundo 40 milhões de pessoas infectadas pelo HIV (o vírus que causa a AIDS), sendo que as taxas de novas infecções continuam crescendo, principalmente na África, Ásia e Rússia. Nesse cenário de pandemia, uma vacina contra o HIV teria imenso impacto, pois salvaria milhões de vidas. Certamente seria um marco na história planetária e também uma esperança para as populações carentes de tratamento antiviral e de acompanhamento médico. TANURI, A.; FERREIRA JUNIOR, O. C. Vacina contra Aids: desafios e esperanças. Ciência Hoje (44) 26, 2009 (adaptado). Uma vacina eficiente contra o HIV deveria '},
    {'query': 'Cabeçalho: Gerente – Boa tarde. Em que eu posso ajudá-lo? Cliente – Estou interessado em financiamento para compra de veículo. Gerente – Nós dispomos de várias modalidades de crédito. O senhor é nosso cliente? Cliente – Sou Júlio César Fontoura, também sou funcionário do banco. Gerente – Julinho, é você, cara? Aqui é a Helena! Cê tá em Brasília? Pensei que você inda tivesse na agência de Uberlândia! Passa aqui pra gente conversar com calma. BORTONI-RICARDO, S. M. Educação em língua materna. São Paulo: Parábola, 2004 (adaptado). \nEnunciado: Na representação escrita da conversa telefônica entre a gerente do banco e o cliente, observa-se que a maneira de falar da gerente foi alterada de repente devido \nAlternativas:\nA. à adequação de sua fala à conversa com um amigo, caracterizada pela informalidade. \nB. à iniciativa do cliente em se apresentar como funcionário do banco. \nC. ao fato de ambos terem nascido em Uberlândia (Minas Gerais). \nD. à intimidade forçada pelo cliente ao fornecer seu nome completo. \nE. ao seu interesse profissional em financiar o veículo de Júlio. \nResposta:', 'choices': ['à adequação de sua fala à conversa com um amigo, caracterizada pela informalidade. ', 'à iniciativa do cliente em se apresentar como funcionário do banco. ', 'ao fato de ambos terem nascido em Uberlândia (Minas Gerais). ', 'à intimidade forçada pelo cliente ao fornecer seu nome completo. ', 'ao seu interesse profissional em financiar o veículo de Júlio. '], 'gold': 0, 'id': '2009_92', 'exam': '2009', 'contents': 'Gerente – Boa tarde. Em que eu posso ajudá-lo? Cliente – Estou interessado em financiamento para compra de veículo. Gerente – Nós dispomos de várias modalidades de crédito. O senhor é nosso cliente? Cliente – Sou Júlio César Fontoura, também sou funcionário do banco. Gerente – Julinho, é você, cara? Aqui é a Helena! Cê tá em Brasília? Pensei que você inda tivesse na agência de Uberlândia! Passa aqui pra gente conversar com calma. BORTONI-RICARDO, S. M. Educação em língua materna. São Paulo: Parábola, 2004 (adaptado). Na representação escrita da conversa telefônica entre a gerente do banco e o cliente, observa-se que a maneira de falar da gerente foi alterada de repente devido '},
    {'query': 'Cabeçalho: A maioria das pessoas daqui era do campo. Vila Maria é hoje exportadora de trabalhadores. Empresários de Primavera do Leste, Estado de Mato Grosso, procuram o bairro de Vila Maria para conseguir mão de obra. É gente indo distante daqui 300, 400 quilômetros para ir trabalhar, para ganhar sete conto por dia. (Carlito, 43 anos, maranhense, entrevistado em 22/03/98). Ribeiro, H. S. O migrante e a cidade: dilemas e conflitos. Araraquara: Wunderlich, 2001(adaptado). \nEnunciado: O texto retrata um fenômeno vivenciado pela agricultura brasileira nas últimas décadas do século XX, consequência \nAlternativas:\nA. dos impactos sociais da modernização da agricultura. \nB. da recomposição dos salários do trabalhador rural. \nC. da exigência de qualificação do trabalhador rural. \nD. da diminuição da importância da agricultura. \nE. dos processos de desvalorização de áreas rurais. \nResposta:', 'choices': ['dos impactos sociais da modernização da agricultura. ', 'da recomposição dos salários do trabalhador rural. ', 'da exigência de qualificação do trabalhador rural. ', 'da diminuição da importância da agricultura. ', 'dos processos de desvalorização de áreas rurais. '], 'gold': 0, 'id': '2010_02', 'exam': '2010', 'contents': 'A maioria das pessoas daqui era do campo. Vila Maria é hoje exportadora de trabalhadores. Empresários de Primavera do Leste, Estado de Mato Grosso, procuram o bairro de Vila Maria para conseguir mão de obra. É gente indo distante daqui 300, 400 quilômetros para ir trabalhar, para ganhar sete conto por dia. (Carlito, 43 anos, maranhense, entrevistado em 22/03/98). Ribeiro, H. S. O migrante e a cidade: dilemas e conflitos. Araraquara: Wunderlich, 2001(adaptado). O texto retrata um fenômeno vivenciado pela agricultura brasileira nas últimas décadas do século XX, consequência '},
    {'query': 'Cabeçalho: A biosfera, que reúne todos os ambientes onde se desenvolvem os seres vivos, se divide em unidades menores chamadas ecossistemas, que podem ser uma floresta, um deserto e até um lago. Um ecossistema tem múltiplos mecanismos que regulam o número de organismos dentro dele, controlando sua reprodução, crescimento e migrações. DUARTE, M. O guia dos curiosos. São Paulo: Companhia das Letras, 1995. \nEnunciado: Predomina no texto a função da linguagem \nAlternativas:\nA. emotiva, porque o autor expressa seu sentimento em relação à ecologia. \nB. fática, porque o texto testa o funcionamento do canal de comunicação. \nC. poética, porque o texto chama a atenção para os recursos de linguagem. \nD. conativa, porque o texto procura orientar comportamentos do leitor. \nE. referencial, porque o texto trata de noções e informações conceituais. \nResposta:', 'choices': ['emotiva, porque o autor expressa seu sentimento em relação à ecologia. ', 'fática, porque o texto testa o funcionamento do canal de comunicação. ', 'poética, porque o texto chama a atenção para os recursos de linguagem. ', 'conativa, porque o texto procura orientar comportamentos do leitor. ', 'referencial, porque o texto trata de noções e informações conceituais. '], 'gold': 4, 'id': '2010_97', 'exam': '2010', 'contents': 'A biosfera, que reúne todos os ambientes onde se desenvolvem os seres vivos, se divide em unidades menores chamadas ecossistemas, que podem ser uma floresta, um deserto e até um lago. Um ecossistema tem múltiplos mecanismos que regulam o número de organismos dentro dele, controlando sua reprodução, crescimento e migrações. DUARTE, M. O guia dos curiosos. São Paulo: Companhia das Letras, 1995. Predomina no texto a função da linguagem '},
    {'query': 'Cabeçalho: O professor Paulo Saldiva pedala 6 km em 22 minutos de casa para o trabalho, todos os dias. Nunca foi atingido por um carro. Mesmo assim, é vítima diária do trânsito de São Paulo: a cada minuto sobre a bicicleta, seus pulmões são envenenados com 3,3 microgramas de poluição particulada – poeira, fumaça, fuligem, partículas de metal em suspensão, sulfatos, nitratos, carbono, compostos orgânicos e outras substâncias nocivas. ESCOBAR, H. Sem Ar. O Estado de São Paulo. Ago. 2008. \nEnunciado: A população de uma metrópole brasileira que vive nas mesmas condições socioambientais das do professor citado no texto apresentará uma tendência de \nAlternativas:\nA. ampliação da taxa de fecundidade. \nB. diminuição da expectativa de vida. \nC. elevação do crescimento vegetativo. \nD. aumento na participação relativa de idosos. \nE. redução na proporção de jovens na sociedade. \nResposta:', 'choices': ['ampliação da taxa de fecundidade. ', 'diminuição da expectativa de vida. ', 'elevação do crescimento vegetativo. ', 'aumento na participação relativa de idosos. ', 'redução na proporção de jovens na sociedade. '], 'gold': 1, 'id': '2011_12', 'exam': '2011', 'contents': 'O professor Paulo Saldiva pedala 6 km em 22 minutos de casa para o trabalho, todos os dias. Nunca foi atingido por um carro. Mesmo assim, é vítima diária do trânsito de São Paulo: a cada minuto sobre a bicicleta, seus pulmões são envenenados com 3,3 microgramas de poluição particulada – poeira, fumaça, fuligem, partículas de metal em suspensão, sulfatos, nitratos, carbono, compostos orgânicos e outras substâncias nocivas. ESCOBAR, H. Sem Ar. O Estado de São Paulo. Ago. 2008. A população de uma metrópole brasileira que vive nas mesmas condições socioambientais das do professor citado no texto apresentará uma tendência de '},
    {'query': 'Cabeçalho: O tema da velhice foi objeto de estudo de brilhantes filósofos ao longo dos tempos. Um dos melhores livros sobre o assunto foi escrito pelo pensador e orador romano Cícero: A Arte do Envelhecimento. Cícero nota, primeiramente, que todas as idades têm seus encantos e suas dificuldades. E depois aponta para um paradoxo da humanidade. Todos sonhamos ter uma vida longa, o que significa viver mais anos. Quando realizamos a meta, em vez de celebrar o feito, nos atiramos a um estado de melancolia e amargura. Ler as palavras de Cícero sobre envelhecimento pode ajudar a aceitar melhor a passagem do tempo. NOGUEIRA, P. Saúde & Bem-Estar Antienvelhecimento. Época. 28 abr. 2008. \nEnunciado: O autor discute problemas relacionados ao envelhecimento, apresentando argumentos que levam a inferir que seu objetivo é \nAlternativas:\nA. esclarecer que a velhice é inevitável. \nB. contar fatos sobre a arte de envelhecer. \nC. defender a ideia de que a velhice é desagradável. \nD. influenciar o leitor para que lute contra o envelhecimento. \nE. mostrar às pessoas que é possível aceitar, sem angústia, o envelhecimento. \nResposta:', 'choices': ['esclarecer que a velhice é inevitável. ', 'contar fatos sobre a arte de envelhecer. ', 'defender a ideia de que a velhice é desagradável. ', 'influenciar o leitor para que lute contra o envelhecimento. ', 'mostrar às pessoas que é possível aceitar, sem angústia, o envelhecimento. '], 'gold': 4, 'id': '2011_106', 'exam': '2011', 'contents': 'O tema da velhice foi objeto de estudo de brilhantes filósofos ao longo dos tempos. Um dos melhores livros sobre o assunto foi escrito pelo pensador e orador romano Cícero: A Arte do Envelhecimento. Cícero nota, primeiramente, que todas as idades têm seus encantos e suas dificuldades. E depois aponta para um paradoxo da humanidade. Todos sonhamos ter uma vida longa, o que significa viver mais anos. Quando realizamos a meta, em vez de celebrar o feito, nos atiramos a um estado de melancolia e amargura. Ler as palavras de Cícero sobre envelhecimento pode ajudar a aceitar melhor a passagem do tempo. NOGUEIRA, P. Saúde & Bem-Estar Antienvelhecimento. Época. 28 abr. 2008. O autor discute problemas relacionados ao envelhecimento, apresentando argumentos que levam a inferir que seu objetivo é '},
    {'query': 'Cabeçalho: Após o retorno de uma viagem a Minas Gerais, onde Pedro I fora recebido com grande frieza, seus partidários prepararam uma série de manifestações a favor do imperador no Rio de Janeiro, armando fogueiras e luminárias na cidade. Contudo, na noite de 11 de março, tiveram início os conflitos que ficaram conhecidos como a Noite das Garrafadas, durante os quais os “brasileiros” apagavam as fogueiras “portuguesas” e atacavam as casas iluminadas, sendo respondidos com cacos de garrafas jogadas das janelas. VAINFAS, R. (Org.). Dicionário do Brasil Imperial. Rio de Janeiro: Objetiva, 2008 (adaptado). \nEnunciado: Os anos finais do I Reinado (1822-1831) se caracterizaram pelo aumento da tensão política. Nesse sentido, a análise dos episódios descritos em Minas Gerais e no Rio de Janeiro revela \nAlternativas:\nA. estímulos ao racismo. \nB. apoio ao xenofobismo. \nC. críticas ao federalismo. \nD. repúdio ao republicanismo. \nE. questionamentos ao autoritarismo. \nResposta:', 'choices': ['estímulos ao racismo. ', 'apoio ao xenofobismo. ', 'críticas ao federalismo. ', 'repúdio ao republicanismo. ', 'questionamentos ao autoritarismo. '], 'gold': 4, 'id': '2012_07', 'exam': '2012', 'contents': 'Após o retorno de uma viagem a Minas Gerais, onde Pedro I fora recebido com grande frieza, seus partidários prepararam uma série de manifestações a favor do imperador no Rio de Janeiro, armando fogueiras e luminárias na cidade. Contudo, na noite de 11 de março, tiveram início os conflitos que ficaram conhecidos como a Noite das Garrafadas, durante os quais os “brasileiros” apagavam as fogueiras “portuguesas” e atacavam as casas iluminadas, sendo respondidos com cacos de garrafas jogadas das janelas. VAINFAS, R. (Org.). Dicionário do Brasil Imperial. Rio de Janeiro: Objetiva, 2008 (adaptado). Os anos finais do I Reinado (1822-1831) se caracterizaram pelo aumento da tensão política. Nesse sentido, a análise dos episódios descritos em Minas Gerais e no Rio de Janeiro revela '},
    {'query': 'Cabeçalho: O sedutor médio. Vamos juntar Nossas rendas e expectativas de vida querida, o que me dizes? Ter 2, 3 filhos e ser meio felizes? VERISSIMO, L. F. Poesia numa hora dessas?! Rio de Janeiro: Objetiva, 2002. \nEnunciado: No poema “O sedutor médio”, é possível reconhecer a presença de posições críticas \nAlternativas:\nA. nos três primeiros versos, em que “juntar expectativas de vida” significa que, juntos, os cônjuges poderiam viver mais, o que faz do casamento uma convenção benéfica. \nB. na mensagem veiculada pelo poema, em que os valores da sociedade são ironizados, o que é acentuado pelo uso do adjetivo “médio” no título e do advérbio “meio” no verso final. \nC. no verso “e ser meio felizes?”, em que “meio” é sinônimo de metade, ou seja, no casamento, apenas um dos cônjuges se sentiria realizado. \nD. nos dois primeiros versos, em que “juntar rendas” indica que o sujeito poético passa por dificuldades financeiras e almeja os rendimentos da mulher. \nE. no título, em que o adjetivo “médio” qualifica o sujeito poético como desinteressante ao sexo oposto e inábil em termos de conquistas amorosas. \nResposta:', 'choices': ['nos três primeiros versos, em que “juntar expectativas de vida” significa que, juntos, os cônjuges poderiam viver mais, o que faz do casamento uma convenção benéfica. ', 'na mensagem veiculada pelo poema, em que os valores da sociedade são ironizados, o que é acentuado pelo uso do adjetivo “médio” no título e do advérbio “meio” no verso final. ', 'no verso “e ser meio felizes?”, em que “meio” é sinônimo de metade, ou seja, no casamento, apenas um dos cônjuges se sentiria realizado. ', 'nos dois primeiros versos, em que “juntar rendas” indica que o sujeito poético passa por dificuldades financeiras e almeja os rendimentos da mulher. ', 'no título, em que o adjetivo “médio” qualifica o sujeito poético como desinteressante ao sexo oposto e inábil em termos de conquistas amorosas. '], 'gold': 1, 'id': '2012_102', 'exam': '2012', 'contents': 'O sedutor médio Vamos juntar Nossas rendas e expectativas de vida querida, o que me dizes? Ter 2, 3 filhos e ser meio felizes? VERISSIMO, L. F. Poesia numa hora dessas?! Rio de Janeiro: Objetiva, 2002. No poema O sedutor médio, é possível reconhecer a presença de posições críticas '},
    {'query': 'Cabeçalho: As Brigadas Internacionais foram unidades de combatentes formadas por voluntários de 53 nacionalidades dispostos a lutar em defesa da República espanhola. Estima-se que cerca de 60 mil cidadãos de várias partes do mundo – incluindo 40 brasileiros – tenham se incorporado a essas unidades. Apesar de coordenadas pelos comunistas, as Brigadas contaram com membros socialistas, liberais e de outras correntes político-ideológicas. SOUZA, I. I. A Guerra Civil Europeia. História Viva, n. 70, 2009 (fragmento). \nEnunciado: A Guerra Civil Espanhola expressou as disputas em curso na Europa na década de 1930. A perspectiva política comum que promoveu a mobilização descrita foi o(a) \nAlternativas:\nA. crítica ao stalinismo. \nB. combate ao fascismo. \nC. rejeição ao federalismo. \nD. apoio ao corporativismo. \nE. adesão ao anarquismo. \nResposta:', 'choices': ['crítica ao stalinismo. ', 'combate ao fascismo. ', 'rejeição ao federalismo. ', 'apoio ao corporativismo. ', 'adesão ao anarquismo. '], 'gold': 1, 'id': '2013_08', 'exam': '2013', 'contents': 'As Brigadas Internacionais foram unidades de combatentes formadas por voluntários de 53 nacionalidades dispostos a lutar em defesa da República espanhola. Estima-se que cerca de 60 mil cidadãos de várias partes do mundo – incluindo 40 brasileiros – tenham se incorporado a essas unidades. Apesar de coordenadas pelos comunistas, as Brigadas contaram com membros socialistas, liberais e de outras correntes político-ideológicas. SOUZA, I. I. A Guerra Civil Europeia. História Viva, n. 70, 2009 (fragmento). A Guerra Civil Espanhola expressou as disputas em curso na Europa na década de 1930. A perspectiva política comum que promoveu a mobilização descrita foi o(a) '},
    {'query': 'Cabeçalho: Própria dos festejos juninos, a quadrilha nasceu como dança aristocrática, oriunda dos salões franceses, depois difundida por toda a Europa. No Brasil, foi introduzida como dança de salão e, por sua vez, apropriada e adaptada pelo gosto popular. Para sua ocorrência, é importante a presença de um mestre “marcante” ou “marcador”, pois é quem determina as figurações diversas que os dançadores desenvolvem. Observa-se a constância das seguintes marcações: “Tour”, “En avant”, “Chez des dames”, “Chez des chevaliê”, “Cestinha de flor”, “Balancê”, “Caminho da roça”, “Olha a chuva”, “Garranchê”, “Passeio”, “Coroa de flores”, “Coroa de espinhos” etc. No Rio de Janeiro, em contexto urbano, apresenta transformações: surgem novas figurações, o francês aportuguesado inexiste, o uso de gravações substitui a música ao vivo, além do aspecto de competição, que sustenta os festivais de quadrilha, promovidos por órgãos de turismo. CASCUDO, L. C. Dicionário do folclore brasileiro. Rio de Janeiro: Melhoramentos, 1976. \nEnunciado: As diversas formas de dança são demonstrações da diversidade cultural do nosso país. Entre elas, a quadrilha é considerada uma dança folclórica por \nAlternativas:\nA. possuir como característica principal os atributos divinos e religiosos e, por isso, identificar uma nação ou região. \nB. abordar as tradições e costumes de determinados povos ou regiões distintas de uma mesma nação. \nC. apresentar cunho artístico e técnicas apuradas, sendo, também, considerada dança-espetáculo. \nD. necessitar de vestuário específico para a sua prática, o qual define seu país de origem. \nE. acontecer em salões e festas e ser influenciada por diversos gêneros musicais. \nResposta:', 'choices': ['possuir como característica principal os atributos divinos e religiosos e, por isso, identificar uma nação ou região. ', 'abordar as tradições e costumes de determinados povos ou regiões distintas de uma mesma nação. ', 'apresentar cunho artístico e técnicas apuradas, sendo, também, considerada dança-espetáculo. ', 'necessitar de vestuário específico para a sua prática, o qual define seu país de origem. ', 'acontecer em salões e festas e ser influenciada por diversos gêneros musicais. '], 'gold': 1, 'id': '2013_108', 'exam': '2013', 'contents': 'Própria dos festejos juninos, a quadrilha nasceu como dança aristocrática, oriunda dos salões franceses, depois difundida por toda a Europa. No Brasil, foi introduzida como dança de salão e, por sua vez, apropriada e adaptada pelo gosto popular. Para sua ocorrência, é importante a presença de um mestre “marcante” ou “marcador”, pois é quem determina as figurações diversas que os dançadores desenvolvem. Observa-se a constância das seguintes marcações: “Tour”, “En avant”, “Chez des dames”, “Chez des chevaliê”, “Cestinha de flor”, “Balancê”, “Caminho da roça”, “Olha a chuva”, “Garranchê”, “Passeio”, “Coroa de flores”, “Coroa de espinhos” etc. No Rio de Janeiro, em contexto urbano, apresenta transformações: surgem novas figurações, o francês aportuguesado inexiste, o uso de gravações substitui a música ao vivo, além do aspecto de competição, que sustenta os festivais de quadrilha, promovidos por órgãos de turismo. CASCUDO, L. C. Dicionário do folclore brasileiro. Rio de Janeiro: Melhoramentos, 1976. As diversas formas de dança são demonstrações da diversidade cultural do nosso país. Entre elas, a quadrilha é considerada uma dança folclórica por '},
    {'query': 'Cabeçalho: É o caráter radical do que se procura que exige a radicalização do próprio processo de busca. Se todo o espaço for ocupado pela dúvida, qualquer certeza que aparecer a partir daí terá sido de alguma forma gerada pela própria dúvida, e não será seguramente nenhuma daquelas que foram anteriormente varridas por essa mesma dúvida. SILVA, F. L. Descartes: a metafísica da modernidade. São Paulo: Moderna, 2001 (adaptado). \nEnunciado: Apesar de questionar os conceitos da tradição, a dúvida radical da filosofia cartesiana tem caráter positivo por contribuir para o(a) \nAlternativas:\nA. dissolução do saber científico. \nB. recuperação dos antigos juízos. \nC. exaltação do pensamento clássico. \nD. surgimento do conhecimento inabalável. \nE. fortalecimento dos preconceitos religiosos. \nResposta:', 'choices': ['dissolução do saber científico. ', 'recuperação dos antigos juízos. ', 'exaltação do pensamento clássico. ', 'surgimento do conhecimento inabalável. ', 'fortalecimento dos preconceitos religiosos. '], 'gold': 3, 'id': '2014_04', 'exam': '2014', 'contents': 'É o caráter radical do que se procura que exige a radicalização do próprio processo de busca. Se todo o espaço for ocupado pela dúvida, qualquer certeza que aparecer a partir daí terá sido de alguma forma gerada pela própria dúvida, e não será seguramente nenhuma daquelas que foram anteriormente varridas por essa mesma dúvida. SILVA, F. L. Descartes: a metafísica da modernidade. São Paulo: Moderna, 2001 (adaptado). Apesar de questionar os conceitos da tradição, a dúvida radical da filosofia cartesiana tem caráter positivo por contribuir para o(a) '},
    {'query': 'Cabeçalho: Por onde houve colonização portuguesa, a música popular se desenvolveu basicamente com o mesmo instrumental. Podemos ver cavaquinho e violão atuarem juntos aqui, em Cabo Verde, em Jacarta, na Indonésia, ou em Goa. O caráter nostálgico, sentimental, é outro ponto comum da música das colônias portuguesas em todo o mundo. O kronjong, a música típica de Jacarta, é uma espécie de lundu mais lento, tocado comumente com flauta, cavaquinho e violão. Em Goa não é muito diferente. De acordo com o texto de Henrique Cazes, grande parte da música popular desenvolvida nos países colonizados por Portugal compartilham um instrumental, destacando-se o cavaquinho e o violão. \nEnunciado: No Brasil, são exemplos de música popular que empregam esses mesmos instrumentos: \nAlternativas:\nA. Maracatu e ciranda. \nB. Carimbó e baião. \nC. Choro e samba. \nD. Chula e siriri. \nE. Xote e frevo. \nResposta:', 'choices': ['Maracatu e ciranda. ', 'Carimbó e baião. ', 'Choro e samba. ', 'Chula e siriri. ', 'Xote e frevo. '], 'gold': 2, 'id': '2014_104', 'exam': '2014', 'contents': 'Por onde houve colonização portuguesa, a música popular se desenvolveu basicamente com o mesmo instrumental. Podemos ver cavaquinho e violão atuarem juntos aqui, em Cabo Verde, em Jacarta, na Indonésia, ou em Goa. O caráter nostálgico, sentimental, é outro ponto comum da música das colônias portuguesas em todo o mundo. O kronjong, a música típica de Jacarta, é uma espécie de lundu mais lento, tocado comumente com flauta, cavaquinho e violão. Em Goa não é muito diferente. De acordo com o texto de Henrique Cazes, grande parte da música popular desenvolvida nos países colonizados por Portugal compartilham um instrumental, destacando-se o cavaquinho e o violão. No Brasil, são exemplos de música popular que empregam esses mesmos instrumentos: '},
    {'query': 'Cabeçalho: Em 1881, a Câmara dos Deputados aprovou uma reforma na lei eleitoral brasileira, a fim de introduzir o voto direto. A grande novidade, porém, ficou por conta da exigência de que os eleitores soubessem ler e escrever. As consequências logo se refletiram nas estatísticas. Em 1872, havia mais de 1 milhão de votantes, já em 1886, pouco mais de 100 mil cidadãos participaram das eleições parlamentares. Houve um corte de quase 90 por cento do eleitorado. CARVALHO, J. M. Cidadania no Brasil: o longo caminho. Rio de Janeiro: Civilização Brasileira, 2006 (adaptado). \nEnunciado: Nas últimas décadas do século XIX, o Império do Brasil passou por transformações como as descritas, que representaram a \nAlternativas:\nA. ascensão dos “homens bons”. \nB. restrição dos direitos políticos. \nC. superação dos currais eleitorais. \nD. afirmação do eleitorado monarquista. \nE. ampliação da representação popular. \nResposta:', 'choices': ['ascensão dos “homens bons”. ', 'restrição dos direitos políticos. ', 'superação dos currais eleitorais. ', 'afirmação do eleitorado monarquista. ', 'ampliação da representação popular. '], 'gold': 1, 'id': '2015_01', 'exam': '2015', 'contents': 'Em 1881, a Câmara dos Deputados aprovou uma reforma na lei eleitoral brasileira, a fim de introduzir o voto direto. A grande novidade, porém, ficou por conta da exigência de que os eleitores soubessem ler e escrever. As consequências logo se refletiram nas estatísticas. Em 1872, havia mais de 1 milhão de votantes, já em 1886, pouco mais de 100 mil cidadãos participaram das eleições parlamentares. Houve um corte de quase 90 por cento do eleitorado. CARVALHO, J. M. Cidadania no Brasil: o longo caminho. Rio de Janeiro: Civilização Brasileira, 2006 (adaptado). Nas últimas décadas do século XIX, o Império do Brasil passou por transformações como as descritas, que representaram a '},
    {'query': 'Cabeçalho: A dança moderna propõe em primeiro lugar o conhecimento de si e o autodomínio. Minha proposta é esta: através do conhecimento e do autodomínio chego à forma, à minha forma — e não o contrário. É uma inversão que muda toda a estética, toda a razão do movimento. A técnica na dança tem apenas uma finalidade: preparar o corpo para responder à exigência do espírito artístico. VIANNA, K.; CARVALHO, M. A. A dança. São Paulo: Siciliano, 1990. \nEnunciado: Na abordagem dos autores, a técnica, o autodomínio e o conhecimento do bailarino estão a serviço da \nAlternativas:\nA. padronização do movimento da dança. \nB. subordinação do corpo a um padrão. \nC. concretização da criação pessoal. \nD. ideia preconcebida de forma. \nE. busca pela igualdade entre os bailarinos. \nResposta:', 'choices': ['padronização do movimento da dança. ', 'subordinação do corpo a um padrão. ', 'concretização da criação pessoal. ', 'ideia preconcebida de forma. ', 'busca pela igualdade entre os bailarinos. '], 'gold': 2, 'id': '2015_97', 'exam': '2015', 'contents': 'A dança moderna propõe em primeiro lugar o conhecimento de si e o autodomínio. Minha proposta é esta: através do conhecimento e do autodomínio chego à forma, à minha forma — e não o contrário. É uma inversão que muda toda a estética, toda a razão do movimento. A técnica na dança tem apenas uma finalidade: preparar o corpo para responder à exigência do espírito artístico. VIANNA, K.; CARVALHO, M. A. A dança. São Paulo: Siciliano, 1990. Na abordagem dos autores, a técnica, o autodomínio e o conhecimento do bailarino estão a serviço da '},
    {'query': 'Cabeçalho: O Rio de Janeiro tem projeção imediata no próprio estado e no Espírito Santo, em parcela do sul do estado da Bahia, e na Zona da Mata, em Minas Gerais, onde tem influência dividida com Belo Horizonte. Compõem a rede urbana do Rio de Janeiro, entre outras cidades: Vitória, Juiz de Fora, Cachoeiro de Itapemirim, Campos dos Goytacazes, Volta Redonda - Barra Mansa, Teixeira de Freitas, Angra dos Reis e Teresópolis. Disponível em: http://ibge.gov.br. Acesso em: 9 jul. 2015 (adaptado). \nEnunciado: O conceito que expressa a relação entre o espaço apresentado e a cidade do Rio de Janeiro é: \nAlternativas:\nA. Frente pioneira. \nB. Zona de transição. \nC. Região polarizada. \nD. Área de conurbação. \nE. Periferia metropolitana. \nResposta:', 'choices': ['Frente pioneira. ', 'Zona de transição. ', 'Região polarizada. ', 'Área de conurbação. ', 'Periferia metropolitana. '], 'gold': 2, 'id': '2016_06', 'exam': '2016', 'contents': 'O Rio de Janeiro tem projeção imediata no próprio estado e no Espírito Santo, em parcela do sul do estado da Bahia, e na Zona da Mata, em Minas Gerais, onde tem influência dividida com Belo Horizonte. Compõem a rede urbana do Rio de Janeiro, entre outras cidades: Vitória, Juiz de Fora, Cachoeiro de Itapemirim, Campos dos Goytacazes, Volta Redonda - Barra Mansa, Teixeira de Freitas, Angra dos Reis e Teresópolis. Disponível em: http://ibge.gov.br. Acesso em: 9 jul. 2015 (adaptado). O conceito que expressa a relação entre o espaço apresentado e a cidade do Rio de Janeiro é: '},
    {'query': 'Cabeçalho: Ler não é decifrar, como num jogo de adivinhações, o sentido de um texto. É, a partir do texto, ser capaz de atribuir-lhe significado, conseguir relacioná-lo a todos os outros textos significativos para cada um, reconhecer nele o tipo de leitura que o seu autor pretendia e, dono da própria vontade, entregar-se a essa leitura, ou rebelar-se contra ela, propondo uma outra não prevista. LAJOLO, M. Do mundo da leitura para a leitura do mundo. São Paulo: Ática, 1993. \nEnunciado: Nesse texto, a autora apresenta reflexões sobre o processo de produção de sentidos, valendo-se da metalinguagem. Essa função da linguagem torna-se evidente pelo fato de o texto \nAlternativas:\nA. ressaltar a importância da intertextualidade. \nB. propor leituras diferentes das previsíveis. \nC. apresentar o ponto de vista da autora. \nD. discorrer sobre o ato de leitura. \nE. focar a participação do leitor. \nResposta:', 'choices': ['ressaltar a importância da intertextualidade. ', 'propor leituras diferentes das previsíveis. ', 'apresentar o ponto de vista da autora. ', 'discorrer sobre o ato de leitura. ', 'focar a participação do leitor. '], 'gold': 3, 'id': '2016_96', 'exam': '2016', 'contents': 'Ler não é decifrar, como num jogo de adivinhações, o sentido de um texto. É, a partir do texto, ser capaz de atribuir-lhe significado, conseguir relacioná-lo a todos os outros textos significativos para cada um, reconhecer nele o tipo de leitura que o seu autor pretendia e, dono da própria vontade, entregar-se a essa leitura, ou rebelar-se contra ela, propondo uma outra não prevista. LAJOLO, M. Do mundo da leitura para a leitura do mundo. São Paulo: Ática, 1993. Nesse texto, a autora apresenta reflexões sobre o processo de produção de sentidos, valendo-se da metalinguagem. Essa função da linguagem torna-se evidente pelo fato de o texto '},
    {'query': 'Cabeçalho: A característica fundamental é que ele não é mais somente um agricultor ou um pecuarista: ele combina atividades agropecuárias com outras atividades não agrícolas dentro ou fora de seu estabelecimento, tanto nos ramos tradicionais urbano-industriais como nas novas atividades que vêm se desenvolvendo no meio rural, como lazer, turismo, conservação da natureza, moradia e prestação de serviços pessoais. SILVA, J. G. O novo rural brasileiro. Revista Nova Economia, n. 1, maio 1997 (adaptado). \nEnunciado: Essa nova forma de organização social do trabalho é denominada \nAlternativas:\nA. terceirização. \nB. pluriatividade. \nC. agronegócio. \nD. cooperativismo. \nE. associativismo. \nResposta:', 'choices': ['terceirização. ', 'pluriatividade. ', 'agronegócio. ', 'cooperativismo. ', 'associativismo. '], 'gold': 1, 'id': '2016_2__02', 'exam': '2016_2_', 'contents': 'A característica fundamental é que ele não é mais somente um agricultor ou um pecuarista: ele combina atividades agropecuárias com outras atividades não agrícolas dentro ou fora de seu estabelecimento, tanto nos ramos tradicionais urbano-industriais como nas novas atividades que vêm se desenvolvendo no meio rural, como lazer, turismo, conservação da natureza, moradia e prestação de serviços pessoais. SILVA, J. G. O novo rural brasileiro. Revista Nova Economia, n. 1, maio 1997 (adaptado). Essa nova forma de organização social do trabalho é denominada '},
    {'query': 'Cabeçalho: A perda de massa muscular é comum com a idade, porém, é na faixa dos 60 anos que ela se torna clinicamente perceptível e suas consequências começam a incomodar no dia a dia, quando simples atos de subir escadas ou ir à padaria se tornam sacrifícios. Esse processo tem nome: sarcopenia. Essa condição ocasiona a perda da força e qualidade dos músculos e tem um impacto significante na saúde. Disponível em: www.infoescola.com. Acesso em: 19 dez. 2012 (adaptado). \nEnunciado: A sarcopenia é inerente ao envelhecimento, mas seu quadro e consequentes danos podem ser retardados com a prática de exercícios físicos, cujos resultados mais rápidos são alcançados com o(a) \nAlternativas:\nA. hidroginástica. \nB. alongamento. \nC. musculação. \nD. corrida. \nE. dança. \nResposta:', 'choices': ['hidroginástica. ', 'alongamento. ', 'musculação. ', 'corrida. ', 'dança. '], 'gold': 2, 'id': '2016_2__103', 'exam': '2016_2_', 'contents': 'A perda de massa muscular é comum com a idade, porém, é na faixa dos 60 anos que ela se torna clinicamente perceptível e suas consequências começam a incomodar no dia a dia, quando simples atos de subir escadas ou ir à padaria se tornam sacrifícios. Esse processo tem nome: sarcopenia. Essa condição ocasiona a perda da força e qualidade dos músculos e tem um impacto significante na saúde. Disponível em: www.infoescola.com. Acesso em: 19 dez. 2012 (adaptado). A sarcopenia é inerente ao envelhecimento, mas seu quadro e consequentes danos podem ser retardados com a prática de exercícios físicos, cujos resultados mais rápidos são alcançados com o(a) '},
    {'query': 'Cabeçalho: Essas moças tinham o vezo de afirmar o contrário do que desejavam. Notei a singularidade quando principiaram a elogiar o meu paletó cor de macaco. Examinavam-no sérias, achavam o pano e os aviamentos de qualidade superior, o feitio admirável. Envaideci-me: nunca havia reparado em tais vantagens. Mas os gabos se prolongaram, trouxeram-me desconfiança. Percebi afinal que elas zombavam e não me susceptibilizei. Longe disso: achei curiosa aquela maneira de falar pelo avesso, diferente das grosserias a que me habituara. Em geral me diziam com franqueza que a roupa não me assentava no corpo, sobrava nos sovacos. RAMOS, G. Infância. Rio de Janeiro: Record, 1994. \nEnunciado: Por meio de recursos linguísticos, os textos mobilizam estratégias para introduzir e retomar ideias, promovendo a progressão do tema. No fragmento transcrito, um novo aspecto do tema é introduzido pela expressão \nAlternativas:\nA. “a singularidade”. \nB. “tais vantagens”. \nC. “os gabos”. \nD. “Longe disso”. \nE. “Em geral”. \nResposta:', 'choices': ['“a singularidade”. ', '“tais vantagens”. ', '“os gabos”. ', '“Longe disso”. ', '“Em geral”. '], 'gold': 3, 'id': '2017_16', 'exam': '2017', 'contents': 'Essas moças tinham o vezo de afirmar o contrário do que desejavam. Notei a singularidade quando principiaram a elogiar o meu paletó cor de macaco. Examinavam-no sérias, achavam o pano e os aviamentos de qualidade superior, o feitio admirável. Envaideci-me: nunca havia reparado em tais vantagens. Mas os gabos se prolongaram, trouxeram-me desconfiança. Percebi afinal que elas zombavam e não me susceptibilizei. Longe disso: achei curiosa aquela maneira de falar pelo avesso, diferente das grosserias a que me habituara. Em geral me diziam com franqueza que a roupa não me assentava no corpo, sobrava nos sovacos. RAMOS, G. Infância. Rio de Janeiro: Record, 1994. Por meio de recursos linguísticos, os textos mobilizam estratégias para introduzir e retomar ideias, promovendo a progressão do tema. No fragmento transcrito, um novo aspecto do tema é introduzido pela expressão '},
    {'query': 'Cabeçalho: Pesquisadores criaram um tipo de plaqueta artificial, feita com um polímero gelatinoso coberto de anticorpos, que promete agilizar o processo de coagulação quando injetada no corpo. Se houver sangramento, esses anticorpos fazem com que a plaqueta mude sua forma e se transforme em uma espécie de rede que gruda nas lesões dos vasos sanguíneos e da pele. MOUTINHO, S. Coagulação acelerada. Disponível em: http://cienciahoje.uol.com.br. Acesso em: 19 fev. 2013 (adaptado). \nEnunciado: Qual a doença cujos pacientes teriam melhora de seu estado de saúde com o uso desse material? \nAlternativas:\nA. Filariose. \nB. Hemofilia. \nC. Aterosclerose. \nD. Doença de Chagas. \nE. Síndrome da imunodeficiência adquirida. \nResposta:', 'choices': ['Filariose. ', 'Hemofilia. ', 'Aterosclerose. ', 'Doença de Chagas. ', 'Síndrome da imunodeficiência adquirida. '], 'gold': 1, 'id': '2017_94', 'exam': '2017', 'contents': 'Pesquisadores criaram um tipo de plaqueta artificial, feita com um polímero gelatinoso coberto de anticorpos, que promete agilizar o processo de coagulação quando injetada no corpo. Se houver sangramento, esses anticorpos fazem com que a plaqueta mude sua forma e se transforme em uma espécie de rede que gruda nas lesões dos vasos sanguíneos e da pele. MOUTINHO, S. Coagulação acelerada. Disponível em: http://cienciahoje.uol.com.br. Acesso em: 19 fev. 2013 (adaptado). Qual a doença cujos pacientes teriam melhora de seu estado de saúde com o uso desse material? '},
]


class ENEM(MultipleChoicePromptSelectionTask):
    VERSION = 0
    DATASET_PATH = 'data/enem'
    DATASET_NAME = None

    KEYS_TO_INDEX = ['context', 'question']
    SEARCHER_K = 10

    use_just_linguistic_and_humanities = False
    tag = None

    manual_examples = manual_examples

    # Note: the stats 'EK_only' and 'TC_only' are valid only for use_just_linguistic_and_humanities=True
    enem_stats = {
        '2009-1':    {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2009-2':    {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
        '2010-1':    {'EK_only': 13, 'TC_only': 16, 'total': 45},
        '2010-2':    {'EK_only': 3, 'TC_only': 25, 'total': 40},
        '2011-1':    {'EK_only': 11, 'TC_only': 12, 'total': 45},
        '2011-2':    {'EK_only': 2, 'TC_only': 21, 'total': 40},
        '2012-1':    {'EK_only': 9, 'TC_only': 21, 'total': 45},
        '2012-2':    {'EK_only': 3, 'TC_only': 23, 'total': 40},
        '2013-1':    {'EK_only': 5, 'TC_only': 19, 'total': 45},
        '2013-2':    {'EK_only': 0, 'TC_only': 23, 'total': 40},
        '2014-1':    {'EK_only': 7, 'TC_only': 13, 'total': 45},
        '2014-2':    {'EK_only': 3, 'TC_only': 22, 'total': 40},
        '2015-1':    {'EK_only': 4, 'TC_only': 22, 'total': 45},
        '2015-2':    {'EK_only': 1, 'TC_only': 23, 'total': 40},
        '2016-1':    {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2016-2':    {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
        '2016_2_-1': {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2016_2_-2': {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
        '2017-1':    {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2017-2':    {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
    }

    # list of top-10 largest documents (> 600 tokens), that will not be used as prompt
    too_large = ['2013_59', '2010_121', '2009_132', '2009_133', '2016_2__104', '2009_130', '2015_108', '2009_131', '2011_128', '2014_135']

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        # download and unpack the dataset
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH, exist_ok=True)
            URL = "https://www.ime.usp.br/~ddm/project/enem/ENEMdataset.zip"
            http_response = urlopen(URL)
            zipfile = ZipFile(BytesIO(http_response.read()))
            zipfile.extractall(path=self.DATASET_PATH)

        self.dataset = collections.defaultdict(list)

        for exam in self.enem_stats:
            if not self.use_just_linguistic_and_humanities:
                n_questions = None
            else:
                n_questions = self.enem_stats[exam]['total']

            # get the documents
            fname = os.path.join(self.DATASET_PATH, exam + '.xml')
            documents = self._parse_xml(exam.split('-')[0], fname, first_n=n_questions, tag=self.tag)

            # Train and test split are the same. However, in fewshot_examples()
            # we ensure the the prompt for each test example will be composed 
            # only with examples from other exams.
            self.dataset['train'] += documents

        self.dataset['train'] = list(map(self._process_doc, self.dataset["train"]))

    def _parse_xml(self, exam, path, tag=None, first_n=None, verbose=True):
        tree = ET.parse(path)
        root = tree.getroot()

        filters = {
            'IC': 'No',
            'MR': 'No',
            'CE': 'No',
        }

        if tag is not None:
            assert tag in ['TC', 'EK', 'DS', 'TC_only', 'EK_only', 'DS_only'], (
                "Please choose 'TC', 'EK', 'DS', 'TC_only', 'EK_only' or 'DS_only'")

            if tag == 'TC':
                filters['TC'] = 'Yes'
            if tag == 'EK':
                filters['EK'] = 'Yes'
            if tag == 'DS':
                filters['DS'] = 'Yes'
            elif tag == 'TC_only':
                filters['TC'] = 'Yes'
                filters['EK'] = 'No'
                filters['DS'] = 'No'
            elif tag == 'EK_only':
                filters['TC'] = 'No'
                filters['EK'] = 'Yes'
                filters['DS'] = 'No'
            elif tag == 'DS_only':
                filters['TC'] = 'No'
                filters['EK'] = 'No'
                filters['DS'] = 'Yes'

        def ignore_question(child, filters):
            for k,v in filters.items():
                if child.get(k) != v:
                    return True
            return False

        documents = []

        for idx, child in enumerate(root):

            if first_n is not None and idx == first_n:
                break

            if ignore_question(child, filters):
                continue

            header = child.find('header').text
            statement = child.find('statement').text

            for pattern, replace in PATTERNS_REPLACES:
                header = apply_regex(pattern, replace, header)
                statement = apply_regex(pattern, replace, statement)
                
            options = []

            answers = child.find('answers')
            for option in answers.iter('option'):
                text = option.text
                for pattern, replace in PATTERNS_REPLACES:
                    if text is not None:
                        text = apply_regex(pattern, replace, text)
                options.append(text)

                if option.get('correct') == 'Yes':
                    correct = option.get('id')

            document = {
                'id': exam + '_' + child.get('id'),  # used to filter out largest prompt candidates
                'exam': exam,  # used to get metrics for each exam, and to filter out prompt candidates
                'context': header,
                'question': statement,
                'options': options,
                'label': correct.lower(),
            }
            assert len(document['options']) == 5, print('The document does not have 5 options')
            documents.append(document)

        return documents

    def create_collection_to_index(self):
        """ Creates a JSON collection to index. Overwrite this funtion to keep
        more arguments.
        """
        with open(self.documents_to_index, 'w') as f:
            json.dump(self.dataset['train'], f)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def test_docs(self):
        return self.dataset["train"]

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
                Passagem: <passage>
                Pergunta: <question>
                Choices:
                A. <choice1>
                B. <choice2>
                C. <choice3>
                D. <choice4>
                Answer:
            """
            prompt = "Cabeçalho: " + doc["context"] + "\n"
            prompt += "Enunciado: " + doc["question"] + "\nAlternativas:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Resposta:"
            return prompt
        choices = ['a', 'b', 'c', 'd', 'e']
        return {
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["label"]),
            "id": doc["id"],
            "exam": doc["exam"],
            "contents": doc["context"] + doc["question"],  # used for indexing
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1. if np.argmax(results) == gold else 0.
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            doc['exam']: acc_norm,
        }
    
    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            '2009': True,
            '2010': True,
            '2011': True,
            '2012': True,
            '2013': True,
            '2014': True,
            '2015': True,
            '2016': True,
            '2016_2_': True,
            '2017': True,
        }
    
    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            '2009': mean,
            '2010': mean,
            '2011': mean,
            '2012': mean,
            '2013': mean,
            '2014': mean,
            '2015': mean,
            '2016': mean,
            '2016_2_': mean,
            '2017': mean,
            "unknown_pred": mean,
        }

    def fewshot_examples(self, k, rnd, prompt_mode, doc):
        # For each doc, limit the self._training_docs to examples from other exams.
        # We also remove the top-10 largest documents from the list of prompt candidates.
        self._training_docs = []
        for d in self.training_docs():
            if d['exam'] != doc['exam'] and d['id'] not in self.too_large:
                self._training_docs.append(d)

        if prompt_mode == 'dynamic-random':
            return rnd.sample(self._training_docs, k)

        elif prompt_mode == 'fixed':
            return rnd.sample(self._training_docs[:k], k)

        elif prompt_mode == 'manual':
            _manual_docs = []
            for d in self.manual_examples:
                if d['exam'] != doc['exam']:
                    _manual_docs.append(d)
            assert k <= len(_manual_docs), (
                f'The number of manual_examples is not enough to satisfy '
                f'num_fewshot={k}. Please, include more examples.')
            return rnd.sample(_manual_docs, k)
            
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

            hits = self.searcher.search(doc['contents'], k=self.SEARCHER_K)
            selected_hits = []

            for hit in hits:
                hit = json.loads(hit.raw)

                if hit['exam'] != doc['exam'] and hit['id'] not in self.too_large:
                    selected_hits.append(hit)

                if len(selected_hits) == k:
                    break

            # check if we have enough similar examples. If not, complete with 
            # random examples.
            i = 0
            while len(selected_hits) < k:
                if self._training_docs[i] not in selected_hits:
                    selected_hits.append(self._training_docs[i])
                i+=1
            
            # move the most relevant examples to the end.
            selected_hits.reverse() 

            return selected_hits

        else:
            print('Please set prompt_mode as "fixed", "dynamic-random", or "dynamic-similar"')


class ENEM_2022(ENEM):

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        super().download(data_dir, cache_dir, download_mode)

        fname = os.path.join(self.DATASET_PATH, '2022.json')
        if os.path.exists(fname):
            print(f"Reusing dataset enem-2022 ({self.DATASET_PATH})")
        else:
            urllib.request.urlretrieve('https://raw.githubusercontent.com/piresramon/gpt-4-enem/main/data/enem/2022.json', fname)

        with open(fname) as f:
            documents = json.load(f)
        
        def ignore_question(doc):
            filters = {
                'IU': False,
                # 'MR': False,  # uncomment to filter out MR
                # 'CE': False,  # uncomment to filter out CE
                'ML': False,
            }
            for k,v in filters.items():
                if doc[k] != v:
                    return True
            return False

        documents = list(filter(lambda doc: not ignore_question(doc), documents))
        self.dataset['test'] = list(map(self._process_doc, documents))

    def process_results(self, doc, results):
        results = super().process_results(doc, results)
        
        # Select the area based on the question-id
        # [  1,  45] languages
        # [ 46,  90] human-sciences
        # [ 91, 135] natural-sciences
        # [126, 190] mathematics
        q_id = int(doc['id'].split('_')[-1])
        area = ['languages', 'human-sciences', 'natural-sciences', 'mathematics'][int(np.ceil(q_id/45))-1]

        results[area] = results['acc']
        return results

    def test_docs(self):
        return self.dataset['test']

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            '2022': True,
            'languages': True,
            'human-sciences': True,
            'natural-sciences': True,
            'mathematics': True,
            'c_languages': True,
            'c_human-sciences': True,
            'c_natural-sciences': True,
            'c_mathematics': True,
        }
    
    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            '2022': mean,
            'languages': mean,
            'human-sciences': mean,
            'natural-sciences': mean,
            'mathematics': mean,
            'c_languages': sum,
            'c_human-sciences': sum,
            'c_natural-sciences': sum,
            'c_mathematics': sum,
            "unknown_pred": mean,
        }


class ENEM_FULL(ENEM):
    VERSION = 0
    DATASET_PATH = 'maritaca-ai/enem'
    DATASET_NAME = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

        # remove annulled questions
        documents = [d for d in self.dataset['train'] if d['label'] in ['A', 'B', 'C', 'D', 'E']]

        self.dataset['test'] = list(map(self._process_doc, documents))

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
                Passagem: <passage>
                Pergunta: <question>
                Choices:
                A. <choice1>
                B. <choice2>
                C. <choice3>
                D. <choice4>
                Answer:
            """
            prompt = doc.get('context', "") + '\n' + doc.get("question", "") + "\nAlternativas:\n"
            prompt = prompt.strip() + '\n'
            alternatives = doc.get('alternatives', doc.get('options'))
            for choice, option in zip(choices, alternatives):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Resposta:"

            # replace [[placeholder]] with the captions
            for desc in doc['description']:
                prompt = prompt.replace('[[placeholder]]', desc, 1)

            return prompt
        
        choices = ['A', 'B', 'C', 'D', 'E']
        return {
            "query": format_example(doc, choices),
            "choices": doc.get('alternatives', doc.get('options')),
            "gold": choices.index(doc["label"].upper()),
            "id": doc["id"],
            "exam": doc["exam"],
            "description": doc.get("description", ""),
            "figures": doc.get("figures", []),
        }

    def process_results(self, doc, results):
        results = super().process_results(doc, results)
        
        # Select the area based on the question-id
        # [  1,  45] languages
        # [ 46,  90] human-sciences
        # [ 91, 135] natural-sciences
        # [126, 190] mathematics
        q_id = int(doc['id'].split('_')[-1])
        area = ['languages', 'human-sciences', 'natural-sciences', 'mathematics'][int(np.ceil(q_id/45))-1]

        results[area] = results['acc']
        return results

    def test_docs(self):
        return self.dataset['test']

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            'languages': True,
            'human-sciences': True,
            'natural-sciences': True,
            'mathematics': True,
            'c_languages': True,
            'c_human-sciences': True,
            'c_natural-sciences': True,
            'c_mathematics': True,
        }
    
    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            'languages': mean,
            'human-sciences': mean,
            'natural-sciences': mean,
            'mathematics': mean,
            'c_languages': sum,
            'c_human-sciences': sum,
            'c_natural-sciences': sum,
            'c_mathematics': sum,
            "unknown_pred": mean,
        }


class ENEM_GREEDY(ENEM):

    def doc_to_target(self, doc):
        return " " + ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']]

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
        gold = ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']]
        pred = results[0]
        unknown=False
        # regex processing. Useful for zero-shot
        match_1 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\.", pred)
        match_2 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDEabcde])\)", pred)
        match_3 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])", pred)
        if len(match_1) > 0:
            pred = match_1[-1] + "."
        elif len(match_2) > 0:
            pred = match_2[-1].upper() + "."
        elif len(match_3) > 0:
            pred = match_3[-1] + "."
        # if the pred matches an alternative text, convert to respective letter
        elif pred in doc['choices']:
            ind = doc['choices'].index(pred)
            pred = ['A.', 'B.', 'C.', 'D.', 'E.'][ind]
        else:
            print(f"Regex failed at processing {pred}")
            print(f'{gold=}, {pred=}, {doc["exam"]=}')
            unknown = True
            pred=""

        acc = 1. if pred == gold else 0.

        return {
            "acc": acc,
            doc['exam']: acc,
            "unknown_pred": 1 if unknown else 0,
            "debug_info":{
                "gold": gold,
                "pred": pred,
            }
        }


class ENEM_2022_GREEDY(ENEM_2022, ENEM_GREEDY):
    pass

class ENEM_FULL_2022_GREEDY(ENEM_FULL, ENEM_GREEDY):
    DATASET_NAME = '2022'

    def process_results(self, doc, results):
        result = super().process_results(doc, results)
        # Remove exam, because we no more compute metric for each exam. 
        # Instead, we use a specific task.
        del result[doc['exam']]  
        return result

class ENEM_FULL_2023_GREEDY(ENEM_FULL, ENEM_GREEDY):
    DATASET_NAME = '2023'

    def process_results(self, doc, results):
        result = super().process_results(doc, results)
        # Remove exam, because we no more compute metric for each exam. 
        # Instead, we use a specific task.
        del result[doc['exam']]
        return result

class ENEM_FULL_2024_GREEDY(ENEM_FULL, ENEM_GREEDY):
    DATASET_NAME = '2024'

    def process_results(self, doc, results):
        result = super().process_results(doc, results)
        # Remove exam, because we no more compute metric for each exam. 
        # Instead, we use a specific task.
        del result[doc['exam']]
        return result