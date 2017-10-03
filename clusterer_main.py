import time
from time import gmtime, strftime
import random
import re
import math
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import collections
import igraph
from typing import List
import sklearn
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity
import skfuzzy as fuzz
from scipy import stats
from pymystem3 import Mystem
from pyaspeller import Word
import gensim
from gensim.models.keyedvectors import KeyedVectors
import hdbscan
from rutermextract import TermExtractor

#one-time download
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

#---------------------------------------------------------------------------

def correctSpelling(line):
    stop_sybols = [u'/', u'\\', u'№', u':', u'1', u'2', u'3', u'4' , u'5', u'6', u'7', u'8', u'9', u'0', u'–']
    checked_words = []
    for word in line.split():
        if not any(st in word for st in stop_sybols):
            try:
                check = Word(word)
                if not check.correct and check.spellsafe:
                    checked_words.append(check.spellsafe.translate({ord(u"-"):" "}))
                else:
                    checked_words.append(word)
            except:
                pass
    return " ".join(checked_words)

def findNounReplacement(stemmer, lemm, pos, word_pos, replace_dict, suffixes):
    lemm_abr = stemmer.stem(lemm)
    stem_difference = 2
    if pos == "VERB":
        lemm_abr = re.sub("ся$", "", lemm_abr)
        stem_difference = 2
        
    if len(lemm_abr) > 4:
        for suff in suffixes:
            if re.search(suff+"$", lemm_abr):
                lemm_abr = re.sub(suff+"$", "", lemm_abr)
                break
        if len(lemm_abr) > 3:
            for word in word_pos:
                if lemm_abr in word:
                    word_abr = stemmer.stem(word)
                    if word_pos[word] == "NOUN" and len(word_abr)-len(lemm_abr) < stem_difference:
                        replace_dict[lemm+"_"+pos] = word+"_"+"NOUN"
                        break

def addReplacement(lemm, word, lemm_abr, word_abr, word_pos, replace_dict, suffixes):
    stem_difference = 2
    if word_pos[word] == "VERB":
        word_abr = re.sub("ся$", "", word_abr)
        stem_difference = 2
        
    for suff in suffixes:
        if re.search(suff+"$", word_abr):
            word_abr = re.sub(suff+"$", "", word_abr)
            break
    if len(word_abr)-len(lemm_abr) < stem_difference:
        replace_dict[word+"_"+word_pos[word]] = lemm+"_"+"NOUN"

def checkWordList(stemmer, lemm, pos, word_pos, replace_dict):
    adj_suff = ['еват','оват','овит','енск','инск','тельн','енн','онн','он','аст','ист','лив','чив','чат','ивн',
                'ск', 'ал','ел','ан','ян','ат','ев','ов','ен','ив','ин','ит','шн','уч','юч','яч','к','л','н']
    verb_suff = ['ствова','ова','ева','чива','ива','ти','нича','ка','а','я','е','и','ну']
    
    if lemm not in word_pos:
        if pos == "ADJ":
            findNounReplacement(stemmer, lemm, pos, word_pos, replace_dict, adj_suff)
        if pos == "VERB":
            findNounReplacement(stemmer, lemm, pos, word_pos, replace_dict, verb_suff)
        if pos == "NOUN":
            lemm_abr = stemmer.stem(lemm)
            lemm_abr = re.sub("(ан)|(ен)$", "", lemm_abr)
            if len(lemm_abr) > 3:          
                for word in word_pos:
                    if lemm_abr in word:
                        word_abr = stemmer.stem(word)
                        if word_pos[word] == "ADJ":
                            addReplacement(lemm, word, lemm_abr, word_abr, word_pos, replace_dict, adj_suff)
                        elif word_pos[word] == "VERB":
                            addReplacement(lemm, word, lemm_abr, word_abr, word_pos, replace_dict, verb_suff)
        word_pos[lemm] = pos

def lemmatize(line, m, stop_list, tags_dict, stemmer, word_pos, replace_dict):
    ana = m.analyze(line)
    lemmas = []
    for item in ana:
        if len(item) > 1 and len(item["analysis"]) > 0:
            lemm = item["analysis"][0]["lex"]
            if len(lemm) > 1 and lemm not in stop_list:
                pos = item["analysis"][0]["gr"].split("=")[0].split(",")[0]
                if pos in tags_dict:
                    pos = tags_dict[pos]
                    checkWordList(stemmer, lemm, pos, word_pos, replace_dict)
                    lemmas.append(lemm+"_"+pos)
    return lemmas

def createIndicatorsDataset(m, stemmer, max_inds_count, stopwords_file, in_file, tags_dict_file, ind_file):
    print("Start creating Indictors DataSet")
    #Загрузка даных
    xl_file = pd.ExcelFile(in_file)
    ds = xl_file.parse("Лист1")
    
    #clear symbols
    # эти символы разбивают слово на два
    chars_to_remove = [u'«', u'»', u'!', u'<', u'>', u'?', u',', u'.', u'-', u'(', u')', u'[', u']', u'"', u'•', u'%', u';']
    dd = {ord(c):" " for c in chars_to_remove}
    dd[ord(u"¬")] = ""
    
    # Загружаем стоп слова
    xl_file = pd.ExcelFile(stopwords_file)
    ds_stop_words = xl_file.parse("Лист1")
    
    stop_list = set()
    for x in ds_stop_words.STOP_WORDS.str.lower().tolist():
        if " " in x:
            for w in x.split():
                stop_list.add(w)
        else:
            stop_list.add(x)
    
    for w in stopwords.words("russian"):
        stop_list.add(w)
        
    print("Кол-во стоп слов: ",len(stop_list))
    
    tags_file = open(tags_dict_file)
    tags_dict = { line.split()[0] : line.split()[1] for line in tags_file }
    tags_file.close()    
    
    # List of indicators
    inds = []
    inds_dict = {}
    #i = 0
    word_pos = {}
    replace_dict = {}
    
    inds_list = [(row[0], str(row[1]).upper()) for row in ds[["IND_ID","IND_NAME"]].values.tolist()]
    #inds_list = ds.IND_NAME.str.upper().tolist()
    if max_inds_count < len(inds_list)*0.9:
        sample_list = random.sample(inds_list, max_inds_count)
    else:
        sample_list = inds_list[0:max_inds_count]
    
    for i, line in sample_list:
        if type(line) == str and len(line) > 0:
            new_line = line.translate(dd)
            new_line = correctSpelling(new_line).upper()      
            if new_line in inds_dict:
                inds.append([i, new_line, inds_dict[new_line], {}])
            else:
                lemmas = lemmatize(new_line, m, stop_list, tags_dict, stemmer, word_pos, replace_dict)
                inds.append([i, new_line, lemmas, {}])
                inds_dict[new_line] = lemmas
                #i += 1
    
    print("Words to replace with nouns >>", len(replace_dict))
    
    for i in range(len(inds)):
        for j in range(len(inds[i][2])):
            if inds[i][2][j] in replace_dict:
                inds[i][2][j] = replace_dict[inds[i][2][j]]
    
    print("indicators >>", inds[0:10])
    
    output = open(ind_file, "wb")
    pickle.dump(inds, output)
    output.close()
    print("Indicators saved in ", ind_file)
    print("\n--------------------------------------------------------\n")

#----------------------------------------------------------------------------------------

def createWordsDataset(m, ind_file, vectors_file, 
                       words_freq_file, words_ds_file, words_dict_file):
    print("Start creating Words DataSet")
    
    pkl_file = open(ind_file, "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    print("Inds", len(inds))
    words_counter = collections.Counter([w for ind in inds for w in ind[2]]) # получили слова с их количествами
    
    # теперь нормируем важность слова относительно
    values = [int(v)  for k,v in dict(words_counter).items()]
    cntr_mean = np.mean(values)
    cntr_std = np.std(values)
    print("cntr_mean >> ", cntr_mean)
    print("cntr_std >> ", cntr_std)
    
    #words_dict = [ [k, (v-cntr_mean)/cntr_std]  for k,v in dict(words_counter).items()]
    words_dict = [ [k, (v-cntr_mean)/cntr_std ] for k,v in dict(words_counter).items() 
                   if v <= (cntr_mean+cntr_std)*5]    

    print("Words cnt: ", len(words_dict))
    print("Words (normalized) [0:10]>> ", words_dict[0:10])
    
    #вывести все слова с частотами в отдельный текстовый файл
    df = pd.DataFrame(words_dict)
    df.to_csv(words_freq_file, index=False, header=True)
    print("Words with frequencies saved in ", words_freq_file)
    
    print("\n--------------------------------------------------------\n")
    
    words_cnt = len(words_dict)
    
    words_ds = [[0 for j in range(0, words_cnt)] for i in range(0, words_cnt)]
    print("Creating words_ds")
    
    # w2v vectors
    model = KeyedVectors.load_word2vec_format(vectors_file, binary=True)
    print("Vectors loaded")
    
    for i in range(0, words_cnt):
        for j in range(0, words_cnt):
            if i==j:
                words_ds[i][j] = 1
                words_ds[j][i] = 1
            else:
                sim = 0
                w1 = words_dict[i][0]
                w2 = words_dict[j][0]
                if w1 in model.vocab and w2 in model.vocab:
                    sim = model.similarity(w1, w2)
                words_ds[i][j] = sim
                words_ds[j][i] = sim
            
    output = open(words_ds_file, "wb")
    pickle.dump(words_ds, output)
    output.close()
    print("Words DS saved in ", words_ds_file)
        
    output = open(words_dict_file, "wb")
    pickle.dump(words_dict, output)
    output.close()
    print("Words dictionary saved in ", words_dict_file)

#----------------------------------------------------------------------------------------
def calcWeigth(vw1, vw2, ew) -> float:
    return ew

def createGraph(words_ds, words, edge_treshold, graph_file_name):
    graph_ver_cnt = len(words)
    g = igraph.Graph()
    g.add_vertices(graph_ver_cnt)
    g.vs["name"] = [k[0] for k in words]
    g.vs["norm_weight"] = [k[1] for k in words]

    edgs = [ (i,j) for i in range(0, graph_ver_cnt) for j in range(0, graph_ver_cnt)
             if i>j and words_ds[i][j] >= edge_treshold]
    g.add_edges(edgs)
    
    g.es["weight"] = [ calcWeigth(words[i][1], words[j][1], words_ds[i][j])
                         for i in range(0, graph_ver_cnt)
                           for j in range(0, graph_ver_cnt) if i>j and words_ds[i][j] > edge_treshold
                     ]
    
    # delete isolated vertices
    exclude_list = []
    exclude_vs = []
    for i in reversed(range(0,graph_ver_cnt)):
        if g.vs[i].degree() == 0:
            exclude_list.append(g.vs[i]["name"])
            exclude_vs.append(i)

    g.delete_vertices(exclude_vs)

    print("Excluded cnt: ", len(exclude_list))
    print("Exclude list [0:50]: ", exclude_list[0:100])
    g.write_graphml(graph_file_name)
    print("Graph "+graph_file_name+" created.")
    igraph.summary(g)
    return g

def constructGraph(start_th, words_ds_file, words_dict_file, graph_file_name_pref):
    pkl_file = open(words_ds_file, "rb")
    words_ds = pickle.load(pkl_file)
    pkl_file.close()
    print("words_ds len: ", len(words_ds))
    
    pkl_file = open(words_dict_file, "rb")
    words = pickle.load(pkl_file)
    pkl_file.close()
    print("words len: ", len(words))

    for th in [start_th]:
        graph_main = createGraph(words_ds, words, th,
                                    graph_file_name_pref+str(round(th*100))+".graphml"
                                   )

#----------------------------------------------------------------------------------------
# Hierarchical clustering based on fuzzy connectedness

def deleteEdges(graph: igraph.Graph, edge_threshold: float):
    for e in graph.es:
        if e["weight"] < edge_threshold:
            graph.delete_edges(e.index)

def getSubgraphWeight(graph: igraph.Graph) -> float:
    norm_weight = [1 if v <= 0 else v+1 for v in graph.vs["norm_weight"]]
    weight = sum(norm_weight[i] for i in range(0, len(graph.vs)) )

    return weight

def getSubgraphKeywords(graph: igraph.Graph, keyword_cnt: int) -> List[str]:
    degree = graph.degree()
    
    norm_weight = [1 if v <= 0 else v+1 for v in graph.vs["norm_weight"]]
    name = graph.vs["name"]

    dict = {name[i]: degree[i]*norm_weight[i] for i in range(0, len(graph.vs)) }

    return sorted(dict, key=dict.get, reverse=True)[0:min(keyword_cnt, len(graph.vs))]

#возвращает разрезанный граф, всегда больше одного подграфа,
# за исключением случая, когда разрезание противоречит органичениям
# старается получить разбиение с заданным количеством значимых (те которые превратятся в узлы) подграфов
def cutGraph(pGraph: igraph.Graph, edge_threshold: float,
              edge_th_step: float,
              max_edge_threshold: float,
              avg_bouquet: int,
              min_subgraph_coeff: float = 10 #коэффициент при котором субграф добавляется в иерархию
             ) -> [[igraph.Graph], float]:

    prev_sgs_cnt = 1
    prev_sgs =  [pGraph]
    sgs = [pGraph]

    #пока возможно разбиение
    while (edge_threshold<1) and (edge_threshold < max_edge_threshold):
        deleteEdges(pGraph, edge_threshold)
        comps = pGraph.components(mode="STRONG")  #Returns:a VertexClustering object
        sgs = comps.subgraphs()
        sgs_cnt = sum(1 if (getSubgraphWeight(sg) >= min_subgraph_coeff) else 0 for sg in sgs)

        #подходит ли нам такое рзбиение?
        #если разбиаение подошло то выходим из цикла
        if (prev_sgs_cnt == 1) and (sgs_cnt >= avg_bouquet):
            break
        else:
            # единственная ситуация продолжения разбиения: не достигли среднего
            if (prev_sgs_cnt == 1) and (sgs_cnt < avg_bouquet):
                prev_sgs_cnt = sgs_cnt
                prev_sgs = sgs
            else:
                # если отклонение от среднего количества на предыдущем шаге было меньше, то возвращаем его
                if abs(prev_sgs_cnt - avg_bouquet) < abs(sgs_cnt - avg_bouquet):
                    sgs = prev_sgs
                    break
                # достигои идеального количества подграфов
                else:
                    break

        #шаг для следующего разбиения
        edge_threshold += edge_th_step

    return [sgs, edge_threshold]

def addLayer(hier_graph: igraph.Graph, graph: igraph.Graph, parent_vtx: str,
              layer_n: int,
              edge_threshold: float, #уровень с которого нужно начинать разбивать граф (для первого случая это текущий уровень)
              edge_th_step: float,
              max_edge_threshold: float,
              max_layer_cnt: int,
              min_subgraph_coeff: float = 10, #коэффициент при котором субграф добавляется в иерархию
              keywords_cnt: int = 10, #количество ключевых слов для узла
              keyword_coeff: float = 100, #коэффициент значимости первого слова от каждого подграфа
              avg_bouquet: int = 4
            ) -> {str: int}:

    #нарежем граф так, чтобы было нужно число подграфов (получим один подграф, только если разрезать не возможно)
    edge_threshold = edge_threshold
    if layer_n > 1: #первый шаг уже приходит нарезанным, поэтому для него условие ELSE
        [sgs, edge_threshold] = cutGraph(graph, edge_threshold,  edge_th_step, max_edge_threshold, avg_bouquet, min_subgraph_coeff)
    else: #первый шаг особенный
        comps = graph.components(mode="STRONG")  #Returns:a VertexClustering object
        sgs = comps.subgraphs()
        if len(sgs) == 1: #если в самом начале нам дали не разбитый граф, то нужно его тоже разбить
            [sgs, edge_threshold] = cutGraph(graph, edge_threshold+edge_th_step,  edge_th_step, max_edge_threshold, avg_bouquet, min_subgraph_coeff)

    #устанавливаем начальные переменные
    keywords = {}
    prnt_index = len(hier_graph.vs)-1
    node_cnt = 1 # если считать с нуля то будет умножение на ноль и пропадут многие ключевые сслова

    #проходимся по всем подграфам
    for sg in sgs:
        sg_keywords = {}
        # если вес данного подграфа достин собственного узла в иерархии то
        # также у нас теперь не может граф распадаться только на одну вершину, мы этого не допускаем процедурой разрезания
        if (getSubgraphWeight(sg) >= min_subgraph_coeff) and (len(sgs) != 1):
            # Add vertex

            # TODO от такого именования нужно будет уйти на более абстрактное, когда будут ключевые слова сделаны
            if len(sg.vs) > keywords_cnt:
                vrtx_name = "Layer "+str(layer_n)+" "+" ".join(list(random.sample(sg.vs["name"], 3)))
            else:
                vrtx_name = "Layer "+str(layer_n)+" "+" ".join(list(sg.vs["name"]))

            hier_graph.add_vertex(vrtx_name)
            sg_vrtx_indx = len(hier_graph.vs)-1
            node_cnt += 1
            hier_graph.vs["layer"] = ["Layer "+str(edge_threshold) if x is None else x for x in hier_graph.vs["layer"]]
            hier_graph.vs["layer_n"] = [edge_threshold if x is None else x for x in hier_graph.vs["layer_n"]]
            hier_graph.vs["graph"] = [sg if x is None else x for x in hier_graph.vs["graph"]]
            hier_graph.vs["parent_node"] = ["n"+str(prnt_index)  if x is None else x for x in hier_graph.vs["parent_node"]]

            hier_graph.add_edge(parent_vtx, vrtx_name)

            # Recursion
            next_edge_threshold = edge_threshold+edge_th_step
            #Условие входа в рекурсию:
            #создавали узел
            #максимальное число шагов не достигнуто
            if (len(sg.vs)>1) and (layer_n < max_layer_cnt) and (next_edge_threshold<1) and (next_edge_threshold < max_edge_threshold):
                sg_keywords = addLayer(hier_graph, sg, vrtx_name, layer_n+1, next_edge_threshold, edge_th_step, max_edge_threshold,
                                          max_layer_cnt, min_subgraph_coeff, keywords_cnt, keyword_coeff, avg_bouquet)
                i = 0
                for k,v in sg_keywords.items(): #пополним список ключевых слов родительской вершины
                    if i == 0:
                        keywords[k] = v*keyword_coeff
                    else:
                        keywords[k] = v
                    i += 1
            else: # в рекурсию не вошли, значит просто пополняем список ключевых слов родителя
                i = 0
                for w in getSubgraphKeywords(sg, keywords_cnt):
                    if i == 0:
                        keywords[w] = 1*keyword_coeff
                    else:
                        keywords[w] = 1
                    sg_keywords[w] = 1
                    i += 1

            # Для добавленного узла нужно вставить его ключевые слова и количество детей
            words = " ".join(sg_keywords.keys())
            # print(words)
            # print(hier_graph.vs["keywords"][0:5])
            hier_graph.vs["keywords"]  = [words if i == sg_vrtx_indx else hier_graph.vs["keywords"][i]
                                            for i in range(0, len(hier_graph.vs))
                                           ]
            # print(hier_graph.vs["keywords"][0:5])
            hier_graph.vs["child_node_cnt"]  = [len(sg.vs()) if i == sg_vrtx_indx else hier_graph.vs["child_node_cnt"][i]
                                                  for i in range(0, len(hier_graph.vs))
                                                 ]

        # если вес данного подграфа НЕ достин собственного узла в иерархии то просто пополняем ключевые слова родителя
        else:
            # just add keywords
            i = 0
            for w in getSubgraphKeywords(sg, keywords_cnt):
                if i == 0:
                    keywords[w] = 1*keyword_coeff
                else:
                    keywords[w] = 1
                i += 1

    keword_list = sorted(keywords, key=keywords.get, reverse=True)[0:keywords_cnt] # у нас уже столько сколько нужно слов

    return {k: node_cnt for k in keword_list}

def getEdgeStat(pGraph: igraph.Graph) -> [float, float, float]:

    if len(pGraph.es["weight"]) == 0:
        return [1,1,1]

    max_weight = max(w for w in  pGraph.es["weight"])
    min_weight = min(w for w in  pGraph.es["weight"])
    avg_weight = sum(w for w in  pGraph.es["weight"])/len(pGraph.es["weight"])

    return [min_weight, avg_weight, max_weight]

def doGraphHierarchicalClustering(th, max_inds_count, graph_file_name_pref, hier_graph_file):
    #parameters
    th_start = th # старовый уровень уничтоженных ребер
    th_step = 0.005 # шаг рекурсии по уничтожению ребер
    th_max = 0.99 # максимальный уровень до которого уничтожаем ребра
    max_depth = 1000 #максимальная глубина рекурсии
    avg_bouquet = 3 # целевое количество подузлов в дереве
    
    min_subgraph_coeff = 0.9 #коэффициент при котором субграф добавляется в иерархию, если <=1 то все слова будут в субграфе
    keywords_cnt = 10 #количество ключевых слов определяющих узел
    keyword_coeff = 100 # множитель для первого слова в ключевых словах от каждого узла (чтобы один узел не затмил своими словами другие)
    
    #load graph
    graph_main = igraph.Graph().Load(graph_file_name_pref+str(round(th*100))+".graphml")
    print("Main graph summary: ", graph_main.summary())
    
    hier_graph = igraph.Graph() # результирующий иерархический граф
    hier_graph.add_vertices(1)
    hier_graph.vs["name"] = ["_Моногорода_"]
    hier_graph.vs["keywords"] = ["_Моногорода_"]
    hier_graph.vs["layer"] = ["Layer "+str(0)]
    hier_graph.vs["layer_n"] = [0]
    hier_graph.vs["child_node_cnt"] = [len(graph_main.vs)]
    hier_graph.vs["parent_node"] = ["n"]
    hier_graph.vs["graph"] = [graph_main]
    
    layer = 1
    parent_vrtx_name = "_Моногорода_"
    
    mono_dict = addLayer(hier_graph, graph_main, parent_vrtx_name, layer,
                            th_start, th_step, th_max,
                            max_depth,
                            min_subgraph_coeff, keywords_cnt, keyword_coeff,
                            avg_bouquet
                           )
    words = " ".join(mono_dict.keys())
    hier_graph.vs["keywords"]  = [words if i == 0 else hier_graph.vs["keywords"][i]
                                    for i in range(0, len(hier_graph.vs))
                                   ]
    hier_graph.vs["child_cnt"]  = hier_graph.degree()
    
    print("Hier graph summary: ", hier_graph.summary())
    print("Graphs' samples: ", hier_graph.vs["parent_node"][0:15])
    print("Graphs' in hierarchy: ", len(hier_graph.vs["graph"]))
    print("Graphs' samples: ", hier_graph.vs["graph"][0:5])
    print("One graph words sample: ", hier_graph.vs["graph"][0].vs["name"][0:10])

    #Вычислим дополнительные полезные характеристики
    # - суммарный вес слов
    # - мин макс и среднее значение веса ребер
    
    hier_graph.vs["subgraph_weigth"] = [getSubgraphWeight(sg)  for sg in  hier_graph.vs["graph"]]
    sg_edge_stats = [getEdgeStat(sg)  for sg in hier_graph.vs["graph"]]
    
    hier_graph.vs["subgraph_edge_mins"] = [i[0] for i in sg_edge_stats]
    hier_graph.vs["subgraph_edge_avgs"] = [i[1] for i in sg_edge_stats]
    hier_graph.vs["subgraph_edge_maxs"] = [i[2] for i in sg_edge_stats]
    
    hier_graph.write_graphml(hier_graph_file)
    
    print("Graph file: "+hier_graph_file)
    return hier_graph
    
#---------------------------------------------------------------------------
# Indicators fuzzy clustering
# выполняется всегда сразу после предыдущего пункта, т.к. тут не загружается граф иерархический

def doIndicatorsFuzzyClustering(hier_graph, max_inds_count, ind_file, clustering_results_file):
    #load graph
    #hier_graph = igraph.Graph().Load(hier_graph_file)
    
    #load indicators
    pkl_file = open(ind_file, "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    print("Inds len: ", len(inds))
    
    graphs = hier_graph.vs["graph"]
    all_words = graphs[0].vs["name"]
    print("All words >> ", all_words[0:100])
    
    for v in hier_graph.vs:
        index = v.index
        graph = graphs[index]
        for i in inds:
            i_len = sum( [1 for w in i[2] if w in all_words] ) #длина показателя в словах нашего графа
            cnt = sum( [1 for w in i[2] if w in graph.vs["name"] ] )
            if i_len > 0 and cnt > 0:
                pcnt = cnt/i_len #len(i[2])
                i[3][index] = round(pcnt,5)
    
    print("Inds sample [0:10]: ", inds[0:9])
    
    #write indicators
    output = open(clustering_results_file+".pkl", "wb")
    pickle.dump(inds, output)
    output.close()
    
    #write indicators to csv
    df = pd.DataFrame(inds)
    df.to_csv(clustering_results_file+".csv", index=False, header=True)
    print("Indicators clustering results saved in ", clustering_results_file+".csv")
    
#---------------------------------------------------------------------------

# test fuzzy hierarchical clustering results

def getIndicatorsByNode(pNodeIndex: int, pInds: list) -> []:
    ind_list=[]
    for i in pInds:
        pt = i[3].get(pNodeIndex, None)
        if pt != None:
            ind_list.append([i, pt])
    ind_list.sort(key=lambda arg: arg[1],  reverse=True)
    return ind_list

def testHierarchicalClusteringResults(hier_graph_file, clustering_results_file):
    # load graph
    hier_graph = igraph.Graph().Load(hier_graph_file)
    
    #load indicators
    pkl_file = open(clustering_results_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    for n in [596, 600, 601, 594]:
        print("Node ID: ", n, " keywords: ", hier_graph.vs["keywords"][n])
        p_inds = getIndicatorsByNode(n, inds)
        print("Inds cnt: ", len(p_inds))
        print("Inds sample [0:10]:")
        for i in p_inds[0:10]:
            print(i[1]," - ",i[0][0]," - ",i[0][1])
        print("----------------------------------")
    
#---------------------------------------------------------------------------
#Добавление в граф дополнительных свойств, чтобы потом с ним было удобне работат

# Для более содержательного анализа было бы хорошо модифицировать наш граф, добавив туда еще два свойства
# - количесво показателей ассоциированных с вершний
# - общий вес показателей ассоциированных с вершиной

def addGraphProperties(hier_graph_file, clustering_results_file):
    # load graph
    hier_graph = igraph.Graph().Load(hier_graph_file)
    
    #load indicators
    pkl_file = open(clustering_results_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    inds_cnt = []
    ind_weigths = []
    
    for v in hier_graph.vs():
        p_inds = getIndicatorsByNode(v.index, inds)
        inds_cnt.append(len(p_inds))
        ind_weigths.append(sum(i[1] for i in p_inds))
    
    hier_graph.vs["inds_cnt"] = inds_cnt
    hier_graph.vs["inds_weigth"] = ind_weigths
    
    hier_graph.write_graphml(hier_graph_file)
    
    print("Graph file: ", hier_graph_file)
    
#---------------------------------------------------------------------------

# Это тоже не шаг, это вспомогательная процедура позволяющая представить граф в более наглядном виде
# Модифицируем граф превратив его в шар

def deleteVertice(graph: igraph.Graph, vertex_index: int):

    for e in graph.incident(vertex_index):
        for v in graph.es()[e].tuple:
            if (v != vertex_index) and (v != 0):
                path_len_1 = graph.shortest_paths(0, v)
                path_len_2 = graph.shortest_paths(0, vertex_index)
                if path_len_2 < path_len_1: #так как у нас дерево то так можно
                    graph.add_edge(0, v)    
                
    graph.delete_vertices(vertex_index)

def transformToSphereGraph(hier_graph_file, sphere_graph_file):
    # load graph
    hier_graph = igraph.Graph().Load(hier_graph_file)
     
    for v in [90,67,66,65,64,63,62,61,60,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]:
        deleteVertice(hier_graph, v)
          
    # write down the graph
    hier_graph.write_graphml(sphere_graph_file)
    
    print("Graph file: ", sphere_graph_file)

#---------------------------------------------------------------------------
# Тоже не шаг, это отдельная задача
# решим такую задачу: дан набор вершин в графе, нужно оставить только те вершины, которые имеют с ними общие показатели

def getIndicatorRelatedNodesByNodeList(pNodeIndexList: list, pInds: list) -> []:
    ind_list={0}
    for i in pInds:
        nodes = i[3].keys()
        b = 0
        for node_index in pNodeIndexList:
            if node_index in nodes:
                b += 1
        if b > 0:
            for n in nodes:
                ind_list.add(n)
    
    for node_index in pNodeIndexList:
        ind_list.remove(node_index)

    return sorted(list(ind_list), key=int, reverse=True)

def getRelatedIndicatorsByNode(nodes, sphere_graph_file, clustering_results_file, 
                               sphere_graph_by_nodes_file):
    # load graph
    hier_graph = igraph.Graph().Load(sphere_graph_file)
    
    #load indicators
    pkl_file = open(clustering_results_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    related_nodes = getIndicatorRelatedNodesByNodeList([69], inds)
    
    print("Related nodes: ", res)
    
    #удалим остальные узлы из графа
    node_ids = hier_graph.vs["id"]
    
    for i in reversed(range(0,len(node_ids))):
        i_id = int( node_ids[i][1:] )
        if i_id not in related_nodes:
            hier_graph.delete_vertices(i)
    
    # write down the graph
    hier_graph.write_graphml(sphere_graph_by_nodes_file)
    
    print("Graph file: ", sphere_graph_by_nodes_file)

#---------------------------------------------------------------------------
#Шаг 5. Четкая кластеризация показателей (четким и нечетким методами)
# Different experiments

def lognormal(x, mu, sigma):
    return (1 / (sigma*x*math.sqrt(2*math.pi)) )* math.exp(-1/2*((math.log(x)-mu)/sigma)**2)

def clusterIndicators(hier_graph_file, clustering_results_file, 
                          hdbscan_results_file, min_cluster_size):
    # load graph
    hier_graph = igraph.Graph().Load(hier_graph_file)
    
    #load indicators
    pkl_file = open(clustering_results_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    print("\n Vertex cont >>", len(hier_graph.vs))
    
    #feature creation
    
    features = []
    #определим функцию которой будем контролировать влияние узла. Будем считать узлы со слишком малым или слышком большим числом показателей менее значительными
    
    print( hier_graph.vs["inds_cnt"][0:10])

    for i in inds:
        features.append( [ ( lognormal(hier_graph.vs["inds_cnt"][j], 2.7, 1) * 1000 )  
                           if i[3].get(j) is not None else 0
                           for j in range(0, len(hier_graph.vs))
                           if j not in [0]
                           ]
                         )
    
    # HDBSCAN -- algoritm -----------------------------------------------------------
    #clustering

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit_predict(features)    
    
    #clusters indicators
    cl_inds = [ [inds[j][0], inds[j][1], inds[j][2] , clusterer.labels_[j]] for j in range(len(inds))]
    
    # write indicators to csv
    df = pd.DataFrame(cl_inds)
    df.to_csv(hdbscan_results_file+".csv", index=False, header=True, encoding = "utf-8")
    print("Indicators clustering results saved in ", hdbscan_results_file+".csv")
    
    output = open(hdbscan_results_file+".pkl", "wb")
    pickle.dump(cl_inds, output)
    output.close()    
    
#--------------------------------------------------------------------------
def makeAverageVector(vectors, weights = None):
    n_vectors = len(vectors)
    if not weights:
        weights = np.ones(n_vectors)
    avg_vector = np.multiply(vectors[0], weights[0])
    if n_vectors > 1:
        for i in range(1,n_vectors):
            avg_vector = np.add(avg_vector, np.multiply(vectors[i], weights[i]))
    return np.divide(avg_vector, n_vectors)

def getIdf(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def classifyIndicators(hdbscan_results_file, vectors_file, hdbscan_results_classified_file, 
                       noise_output_file=False):
    pkl_file = open(hdbscan_results_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    n_clusters = list(set([ind[3] for ind in inds]))
    n_clusters = n_clusters[:len(n_clusters)-1]
    
    clusters_data = [ [ [ind[0], ind[1], ind[2], ind[3], [] ] for ind in inds if ind[3] == cluster] 
                      for cluster in n_clusters]
    noise_data = [ [ind[0], ind[1], ind[2], ind[3], [] ] for ind in inds if ind[3] == -1]
    
    classified_data = []
    all_inds_lemmas = [ind[2] for cluster in clusters_data for ind in cluster] + [ind[2] for ind in noise_data]
    idf = getIdf(all_inds_lemmas)
    
    model = KeyedVectors.load_word2vec_format(vectors_file, binary=True)
    new_clusters = {}

    for cluster in clusters_data:
        sentence_vectors = []
        ind_vec = {}
        for ind in cluster:
            if len(ind[2]) > 0:
                lemm_vectors = [model[lemm] for lemm in ind[2] if lemm in model.vocab]
                lemm_weights = [idf[lemm] for lemm in ind[2] if lemm in model.vocab]
                if len(lemm_vectors) > 0:
                    sentence_vector = makeAverageVector(lemm_vectors, lemm_weights)
                    sentence_vectors.append(sentence_vector)
                    ind_vec[ind[0]] = sentence_vector
        cluster_vector = makeAverageVector(sentence_vectors)
    
        clean_cluster = []
        clean_ind_vec = []
        for ind in cluster:
            if ind[0] in ind_vec:
                if cosine_similarity([ind_vec[ind[0]]], [cluster_vector])[0] >= 0.8:
                    clean_cluster.append(ind)
                    clean_ind_vec.append(ind_vec[ind[0]])
                else:
                    noise_data.append([ind[0], ind[1], ind[2], -1, []])
        if len(clean_ind_vec) > 0:
            clean_cluster_vector = makeAverageVector(clean_ind_vec)
            
            new_clusters[cluster[0][3]] = [clean_cluster_vector, clean_cluster]
    
    new_n_clusters = list(new_clusters.keys())
    prev_cluster = new_n_clusters[0]
    for cluster in new_n_clusters[1:]:
        sim = cosine_similarity([new_clusters[prev_cluster][0]], [new_clusters[cluster][0]])[0]
        print(prev_cluster, cluster, sim)
        if sim >= 0.85:
            new_clusters[prev_cluster][1] += new_clusters[cluster][1]
            new_clusters[prev_cluster][0] = makeAverageVector([new_clusters[prev_cluster][0], new_clusters[cluster][0]])
            del new_clusters[cluster]
        else:
            prev_cluster = cluster
    
    for cl in new_clusters:
        keywords = [word.split("_")[0] for word, freq in model.most_similar(positive = [new_clusters[cl][0]], topn=10)]
        new_clusters[cl].append(keywords)
        for ind in new_clusters[cl][1]:
            if len(ind[2]) > 0:
                ind[3] = cl
                ind[4] = keywords
                classified_data.append(ind)
            else:
                ind[3] = "-1"
                classified_data.append(ind)                
    
    noise_left = []
    for i, text, lemmas, n, kw in noise_data:
        lemm_vectors = [model[lemm] for lemm in lemmas if lemm in model.vocab]
        lemm_weights = [idf[lemm] for lemm in lemmas if lemm in model.vocab]
        sentence_vector = makeAverageVector(lemm_vectors, lemm_weights)
        
        for cl in new_clusters:
            if cosine_similarity([sentence_vector], [new_clusters[cl][0]])[0] >= 0.9:
                classified_data.append([i, text, lemmas, cl, new_clusters[cl][2]])   #+"+"
                break
            if cl == max(new_clusters):
                noise_left.append([i, text, lemmas, n, kw])
                
    classified_data += noise_left
    df = pd.DataFrame(classified_data)
    df.to_csv(hdbscan_results_classified_file+".csv", index=False, header=True, encoding = "utf-8")
    print("Indicators classification results saved in ", hdbscan_results_classified_file+".csv")
    
    output = open(hdbscan_results_classified_file+".pkl", "wb")
    pickle.dump(classified_data, output)
    output.close()    
    
    if noise_output_file:
        noise_data_output = [ [ind[0], ind[1], ind[2], {}] for ind in noise_left]
        output = open(noise_output_file+".pkl", "wb")
        pickle.dump(noise_data_output, output)
        output.close()
        
#--------------------------------------------------------------------------

def clusterNoise(m, noise_output_file, path, prefix, suffix, start_th, vectors_file, min_cluster_size):
    suffix += "_noise"
    words_freq_file = path+prefix+suffix+".csv"
    words_ds_file = path+prefix+"words_ds_"+suffix+".pkl"
    words_dict_file = path+prefix+"words_"+suffix+".pkl"
    graph_file_name_pref = path+prefix+"words_graph_"+suffix+"_th_"
    hier_graph_file = path+prefix+"words_hier_graph_"+suffix+"_th_"+str(round(start_th*100))+".graphml"
    clustering_results_file = path+prefix+"hier_"+suffix+"_th_"+str(round(start_th*100))
    hdbscan_results_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_hdbscan"
    hdbscan_results_classified_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_hdbscan_classified"
    
    pkl_file = open(noise_output_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    max_inds_count = len(inds)
    
    createWordsDataset(m, noise_output_file+".pkl", vectors_file, words_freq_file, words_ds_file, words_dict_file)

    constructGraph(start_th, words_ds_file, words_dict_file, graph_file_name_pref)

    hier_graph = doGraphHierarchicalClustering(start_th, max_inds_count, graph_file_name_pref, hier_graph_file)

    doIndicatorsFuzzyClustering(hier_graph, max_inds_count, noise_output_file+".pkl", clustering_results_file)

    addGraphProperties(hier_graph_file, clustering_results_file)

    clusterIndicators(hier_graph_file, clustering_results_file, hdbscan_results_file, min_cluster_size)    

    classifyIndicators(hdbscan_results_file, vectors_file, noise_output_file)    

#--------------------------------------------------------------------------

def mergeClusteredNoise(hdbscan_results_classified_file, noise_output_file, merged_file):
    pkl_file = open(hdbscan_results_classified_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close() 
    
    pkl_file = open(noise_output_file+".pkl", "rb")
    inds_noise = pickle.load(pkl_file)
    pkl_file.close()
    
    n_clusters = list(set([ind[3] for ind in inds]))
    n_clusters = n_clusters[:len(n_clusters)-1]
    last_cluster = max(n_clusters)+1
    
    merged_data = [  ind for ind in inds for cluster in n_clusters if ind[3] == cluster]
    
    n_noise_clusters = list(set([ind[3] for ind in inds_noise]))
    n_noise_clusters = n_noise_clusters[:len(n_noise_clusters)-1]
    
    merged_data += [  [ind[0], ind[1], ind[2], ind[3]+last_cluster, ind[4]] for ind in inds_noise 
                      for cluster in n_noise_clusters if ind[3] == cluster]
    
    merged_data += [ ind for ind in inds_noise if ind[3] == -1]
    
    df = pd.DataFrame(merged_data)
    df.to_csv(merged_file+".csv", index=False, header=True, encoding = "utf-8") 
    
    output = open(merged_file+".pkl", "wb")
    pickle.dump(merged_data, output)
    output.close()

#--------------------------------------------------------------------------

def extractKeyterms(term_extractor, merged_file, keyterms_file):
    pkl_file = open(merged_file+".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()
    
    n_clusters = list(set([ind[3] for ind in inds]))
    n_clusters = n_clusters[:len(n_clusters)-1]
    
    clusters_data = [ [ ind for ind in inds if ind[3] == cluster] for cluster in n_clusters]
    noise_data = [ ind + [] for ind in inds if ind[3] == -1]
    
    inds_with_keyterms = []
    
    for cluster in clusters_data:
        term_counter = collections.Counter()
        for ind in cluster:
            sentence = correctSpelling(ind[1].lower())
            term_counter.update([term.normalized for term in term_extractor(sentence)])
    
        keyterms = [term for term in term_counter if term_counter[term]]        
        for ind in cluster:
            inds_with_keyterms.append([ind[0], ind[1], ind[2], ind[3], keyterms, ind[4]])
    
    inds_with_keyterms += noise_data
    
    df = pd.DataFrame(inds_with_keyterms)
    df.to_csv(keyterms_file, index=False, header=True, encoding = "utf-8") 
    print("Cluster keyterms saved in ", keyterms_file)
    
#--------------------------------------------------------------------------


def main():
    m = Mystem()
    stemmer = SnowballStemmer("russian")
    term_extractor = TermExtractor()
    
    path = "med_inds_clustering/"
    max_inds_count = 19035
    prefix = "june_inds_"
    suffix = "med"
    start_th = 0.5  #start threshold for fuzzy graph
    
    in_file = path+"june_inds_med_by_kw.xlsx"
    stopwords_file = path+"Стоп слова Топонимы.xlsx"
    vectors_file = path+"ruwikiruscorpora_0_300_20.bin"
    #vectors_file = path+"wiki.ru.vec" #fasttext
    tags_dict_file = path+"tags_dict.txt"
    
    #ind_file = path+prefix+suffix+".pkl"
    ind_file = path+"june_inds_med_by_kw.pkl"
    words_freq_file = path+prefix+suffix+".csv"
    words_ds_file = path+prefix+"words_ds_"+suffix+".pkl"
    words_dict_file = path+prefix+"words_"+suffix+".pkl"
    graph_file_name_pref = path+prefix+"words_graph_"+suffix+"_th_"
    hier_graph_file = path+prefix+"words_hier_graph_"+suffix+"_th_"+str(round(start_th*100))+".graphml"
    clustering_results_file = path+prefix+"hier_"+suffix+"_th_"+str(round(start_th*100))
    #sphere_graph_file = path+prefix+"words_sphere_graph_"+suffix+"_th_"+str(round(start_th*100))+".graphml"
    #sphere_graph_by_nodes_file = path+prefix+"words_sphere_graph_"+suffix+"_th_"+str(round(start_th*100))+"_69.graphml"
    
    hdbscan_results_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_hdbscan"
    hdbscan_results_classified_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_hdbscan_classified"
    noise_output_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_noise"
    merged_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_hdbscan_classified_merged"
    keyterms_file = path+prefix+suffix+"_th_"+str(round(start_th*100))+"_hdbscan_classified_keyterms.csv"

    print("--- Start time %s ---" % strftime("%a, %d %b %Y %H:%M:%S", gmtime()))
    start_time = time.time()
    """
    createIndicatorsDataset(m, stemmer, max_inds_count, stopwords_file, in_file, tags_dict_file, ind_file)
    
    createWordsDataset(m, ind_file, vectors_file, words_freq_file, words_ds_file, words_dict_file)
    
    constructGraph(start_th, words_ds_file, words_dict_file, graph_file_name_pref)
    
    hier_graph = doGraphHierarchicalClustering(start_th, max_inds_count, graph_file_name_pref, hier_graph_file)
    
    doIndicatorsFuzzyClustering(hier_graph, max_inds_count, ind_file, clustering_results_file)
    
    #testHierarchicalClusteringResults(hier_graph_file, clustering_results_file)
    
    addGraphProperties(hier_graph_file, clustering_results_file)
    
    #transformToSphereGraph(hier_graph_file, sphere_graph_file)
    
    #getRelatedIndicatorsByNode(nodes, sphere_graph_file, clustering_results_file, sphere_graph_by_nodes_file)
    """
    clusterIndicators(hier_graph_file, clustering_results_file, hdbscan_results_file, min_cluster_size = 18)
    
    classifyIndicators(hdbscan_results_file, vectors_file, hdbscan_results_classified_file, noise_output_file)
    
    clusterNoise(m, noise_output_file, path, prefix, suffix, start_th, vectors_file, min_cluster_size = 8)
    
    mergeClusteredNoise(hdbscan_results_classified_file, noise_output_file, merged_file)
    
    extractKeyterms(term_extractor, merged_file, keyterms_file)
    
    print("Done")
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    
if __name__ == "__main__":
    main()
