# import cluster_main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
from time import gmtime, strftime
from sklearn.decomposition import PCA
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import igraph
import pickle
import pandas as pd
import clustering_assessment as ca
import hdbscan
import math
import common_utils as cu
from gensim.models.keyedvectors import KeyedVectors
import experiments_config as exp_conf
# import sklearn.metrics.pairwise as pw

#-----------------------------------------------------------------------------------------------------------------------

# Утилита для получения значений логнормального распределения
def lognormal(x, mu, sigma):
    return (1 / (sigma*x*math.sqrt(2*math.pi)) )* math.exp(-1/2*((math.log(x)-mu)/sigma)**2)

#-----------------------------------------------------------------------------------------------------------------------

def showLogNormal(mu, p_show_mode = 1):
  T = [[],[]]
  for i in range(100):
      x = 0.1+(100-0.1)*i/100
      T[0].append(x)
      T[1].append(100*lognormal(x, mu, 1))

  print("\n LogNorm for ", mu, " >> ", T[1])
  if p_show_mode == 1:
      sns.set_context('poster')
      sns.set_color_codes()
      plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}
      plt.scatter(T[0], T[1], c='b', **plot_kwds)
      frame = plt.gca()
      # frame.axes.get_xaxis().set_visible(False)
      # frame.axes.get_yaxis().set_visible(False)
      plt.show()
  return T[1]


#-----------------------------------------------------------------------------------------------------------------------

def getClusteringTask(hier_graph_file, ind_fuzzy_clustering_results_file, p_target_file, p_target_list):

    print("Start getClusteringTask")
    # load graph
    hier_graph = igraph.Graph().Load(hier_graph_file)

    # load indicators
    pkl_file = open(ind_fuzzy_clustering_results_file + ".pkl", "rb")
    inds = pickle.load(pkl_file)
    pkl_file.close()

    #load target data
    xl_file = pd.ExcelFile(p_target_file)
    ds = xl_file.parse(p_target_list)
    l_target_ind_names = ds["IND_NAME"].values.tolist()
    l_target_ind_clusters = ds["TARGET"].values.tolist()

    # l_target_ind_names = [ i[1] for i in inds]

    # print("\n l_target_ind_names >>", l_target_ind_names[0:10])
    # print("\n l_ind_names >>", inds[0:10][1])
    print("\n Vertex cont >>", len(hier_graph.vs))

    # feature creation

    features_flat = []
    features_lognorm = []
    features_w2v = []
    features_w2v_pos = []
    features_w2v_norm1 = []
    features_w2v_norm2 = []
    features_w2v_norm1_pos = []
    features_w2v_norm2_pos = []

    ind_names = []
    ind_targets = []
    # определим функцию которой будем контролировать влияние узла. Будем считать узлы со слишком малым или слышком большим числом показателей менее значительными

    # print(hier_graph.vs["inds_cnt"][0:10])

    for i in inds:
        if i[1] in l_target_ind_names:
            features_flat.append([ 1+i[3].get(j)
                             if i[3].get(j) is not None else 0
                             for j in range(0, len(hier_graph.vs))
                             if (j not in [0]) and (hier_graph.vs["subgraph_edge_mins"][j] > 0.49)  #and (hier_graph.vs["child_node_cnt"][j] < 100)
                            ]
                           )

            l_ind_vector = model.word_vec(i[2][0])
            l_ind_vector_pos = model.word_vec(i[2][0])

            for k in range(1, len(i[2])):
                if i[2][k] in model.vocab:
                    l_ind_vector = np.add(l_ind_vector, model.word_vec(i[2][k]))
                    if np.core.multiarray.dot(l_ind_vector, model.word_vec(i[2][k])) > 0: #скалярное произведение векторов
                        l_ind_vector_pos = np.add(l_ind_vector_pos, model.word_vec(i[2][k]))
                    else: # ноль можно трактовать как угодно только не 0, поэтому не использую ышпт
                        l_ind_vector_pos = np.add(l_ind_vector_pos, -1.0*model.word_vec(i[2][k]))

            l_sum1 = sum([k for k in l_ind_vector])
            l_sum2 = sum([k * k for k in l_ind_vector])
            l_sum1_pos = sum([k for k in l_ind_vector_pos])
            l_sum2_pos = sum([k * k for k in l_ind_vector_pos])

            features_w2v.append(l_ind_vector)
            features_w2v_pos.append(l_ind_vector_pos)
            features_w2v_norm1.append( [float(i)/l_sum1 for i in l_ind_vector] )
            features_w2v_norm2.append( [float(i)/l_sum2 for i in l_ind_vector] )
            features_w2v_norm1_pos.append( [float(i)/l_sum1_pos for i in l_ind_vector_pos] )
            features_w2v_norm2_pos.append( [float(i)/l_sum2_pos for i in l_ind_vector_pos] )

            #non normalized case
            # l_word_matrix = [model.word_vec(w, use_norm=True)  for w in i[2] if w in model.vocab]
            # l_ind_vector = [sum(wv) for wv in zip(*l_word_matrix)]
            # features_w2v.append(l_ind_vector)

            # square normalization
            # l_sum = sum([k*k  for k in l_ind_vector])
            # l_norm_ind_vector = [float(i)/l_sum for i in l_ind_vector]
            # features_w2v.append(l_norm_ind_vector)


            ind_names.append(i[1])
            ind_targets.append(l_target_ind_clusters[l_target_ind_names.index(i[1])])

    # Была идея что нужно добавить один элемент со всеми нулями чтобы не применялась нормировка алгоритма. На качесвте не отразилось поэтому отказался
    # features_flat.append([ 0
    #                  for j in range(0, len(hier_graph.vs))
    #                  if (j not in [0]) and (hier_graph.vs["subgraph_edge_mins"][j] > 0.49)  #and (hier_graph.vs["child_node_cnt"][j] < 100)
    #                 ]
    #                )
    # ind_targets.append(max(ind_targets)+1)



    features_list = [#features_flat,
                     features_w2v, features_w2v_pos#, features_w2v_norm1, features_w2v_norm2, features_w2v_norm1_pos,
                     #features_w2v_norm2_pos
                    ]
    features_names = [#'Плоские фичи',
                      'word2vec', 'word2vec positive'#, 'word2vec norm l1', 'word2vec norm l2', 'word2vec norm l1 positive', 'word2vec norm l2 positive'
                      ]

    # features_list = [features_flat,
    #                  features_w2v, features_w2v_pos, features_w2v_norm1, features_w2v_norm2, features_w2v_norm1_pos, features_w2v_norm2_pos
    #                 ]
    # features_names = ['Плоские фичи',
    #                   'word2vec', 'word2vec positive', 'word2vec norm l1', 'word2vec norm l2', 'word2vec norm l1 positive', 'word2vec norm l2 positive'
    #                   ]
    # # эксперимент по подброу коэффициентов
    # for k in range(70):
    #     features_list.append([])
    #     features_names.append('lognorm ' + str(1 + k/10))
    #     for i in inds:
    #         if i[1] in l_target_ind_names:
    #             features_list[k].append([  (1+i[3].get(j)) *  (lognormal(hier_graph.vs["inds_cnt"][j], 1 + k/10, 1)*100 ) #* 1000  inds_cnt  i[3].get(j) *  (1+i[3].get(j)) *
    #                              if i[3].get(j) is not None else 0
    #                              for j in range(0, len(hier_graph.vs))
    #                              if (j not in [0]) and (hier_graph.vs["subgraph_edge_mins"][j] > 0.49) #and (hier_graph.vs["child_node_cnt"][j] < 100)
    #                              ]
    #                             )
    #     # features_list[k].append([ 0
    #     #                  for j in range(0, len(hier_graph.vs))
    #     #                  if (j not in [0]) and (hier_graph.vs["subgraph_edge_mins"][j] > 0.49)  #and (hier_graph.vs["child_node_cnt"][j] < 100)
    #     #                 ]
    #     #                )

    return [ind_names, ind_targets, features_names,  features_list]

    print("Finish getClusteringTask")

#-----------------------------------------------------------------------------------------------------------------------

def doClustering(data, algorithm, args, kwds, show_mode = 1):
    print("Start doClustering")

    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()

    if show_mode == 1:
        sns.set_context('poster')
        sns.set_color_codes()
        plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}

        pca = PCA(n_components=2)
        data_ND = pca.fit(data).transform(data)
        # data_2D = data

        palette = sns.color_palette('bright', np.unique(labels).max() + 1)   # deep
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        xs = [i[0] for i in data_ND]
        ys = [i[1] for i in data_ND]

        #  3D variant
        # zs = [i[2] for i in data_ND]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xs, ys, zs, c=colors, marker='o')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # 2D variant
        plt.scatter( xs, ys, c=colors, **plot_kwds)
        frame = plt.gca()
        # frame.axes.get_xaxis().set_visible(False)
        # frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
        plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
        plt.show()

        # print histogram of clusters volume

        # не отображаем мусорный кластер
        plt.hist([l for l in labels if l > -1], bins=np.arange(labels.min(), labels.max() + 1), align='left')

        # отображаем мусорный кластер
        # plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 1), align='left')

        plt.show()

    print("Finish doClustering")
    return labels


#-----------------------------------------------------------------------------------------------------------------------

def performClusteringForFetures(p_features, p_ind_target, p_performe_cosine_dist = 0):

    l_alg_list = []
    l_assess_res_list = []

    # --- K-MEANS EXACT------
    l_labes_kmeans = doClustering(p_features, cluster.KMeans, (), {'n_clusters': max(p_ind_target)+1}, 0)

    # print("\n count l_ind_target >> ", len(p_ind_target))
    # print("\n count l_labes_kmeans >> ", len(l_labes_kmeans))

    # assess clustering
    l_assess_res = ca.getClusteringQuality(l_labes_kmeans, p_ind_target)
    l_assess_res.append(0) #l_garbage_cluster_size
    l_assess_res.append(max(p_ind_target)+1) # l_garbage_cluster_size
    # print("Clustering compare avg precision and accuracy ", l_assess_res)
    l_alg_list.append('K-means '+str(max(p_ind_target)+1))
    l_assess_res_list.append(l_assess_res)

    # --- K-MEANS EXACT------
    l_labes_kmeans = doClustering(p_features, cluster.KMeans, (), {'n_clusters': 5*(max(p_ind_target)+1)}, 0)

    # assess clustering
    l_assess_res = ca.getClusteringQuality(l_labes_kmeans, p_ind_target)
    l_assess_res.append(0) #l_garbage_cluster_size
    l_assess_res.append(5*(max(p_ind_target)+1)) # l_garbage_cluster_size
    l_alg_list.append('K-means 5*exact '+str(5*(max(p_ind_target)+1)))
    l_assess_res_list.append(l_assess_res)

    # --- HDBScan min=17 distance = manhattan------
    l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 17, 'metric':'manhattan'}, 0)  #'eom', 'leaf' 'euclidean' 'metric':'manhattan'   , 'metric':'pyfunc', 'func':cu.cosine_metric
    # assess clustering
    # сначала надо переименовать мусорный -1 кластер в последний какой то
    l_cluster_cnt = l_labes_hdbscan.max()
    print("HDBScan has found N clusters (without noise one)>> ", l_cluster_cnt)
    l_labes_hdbscan = [l_cluster_cnt if i == -1 else i for i in l_labes_hdbscan]
    # print(l_labes_hdbscan[0:100])
    # l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, -1)
    # print("Clustering compare avg precision and accuracy for dbscan", l_assess_res)

    # time.sleep(1.1)  # pause 1.1 seconds  это нужно чтобы логи не перетерлись
    l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, l_cluster_cnt)
    l_garbage_cluster_size = len([i for i in l_labes_hdbscan if i == l_cluster_cnt])
    l_assess_res.append(l_garbage_cluster_size)
    l_assess_res.append(l_cluster_cnt)
    # print("Clustering compare avg precision and accuracy for dbscan without garbage cluster ( indicatores removed) "
    #       , l_garbage_cluster_size
    #       , " indicatores removed) "
    #       , l_assess_res)
    l_alg_list.append('HDBScan min-size=17 distance = manhattan')
    l_assess_res_list.append(l_assess_res)

    # --- HDBScan min=7 distance = manhattan------
    l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 7, 'metric':'manhattan'}, 0)  #'eom', 'leaf' 'euclidean' 'metric':'manhattan'   , 'metric':'pyfunc', 'func':cu.cosine_metric
    # assess clustering
    # сначала надо переименовать мусорный -1 кластер в последний какой то
    l_cluster_cnt = l_labes_hdbscan.max()
    print("HDBScan has found N clusters (without noise one)>> ", l_cluster_cnt)
    l_labes_hdbscan = [l_cluster_cnt if i == -1 else i for i in l_labes_hdbscan]
    # print(l_labes_hdbscan[0:100])
    # l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, -1)
    # print("Clustering compare avg precision and accuracy for dbscan", l_assess_res)

    # time.sleep(1.1)  # pause 1.1 seconds  это нужно чтобы логи не перетерлись
    l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, l_cluster_cnt)
    l_garbage_cluster_size = len([i for i in l_labes_hdbscan if i == l_cluster_cnt])
    l_assess_res.append(l_garbage_cluster_size)
    l_assess_res.append(l_cluster_cnt)
    # print("Clustering compare avg precision and accuracy for dbscan without garbage cluster ( indicatores removed) "
    #       , l_garbage_cluster_size
    #       , " indicatores removed) "
    #       , l_assess_res)
    l_alg_list.append('HDBScan min-size=7 distance = manhattan')
    l_assess_res_list.append(l_assess_res)

    # --- HDBScan min=5 distance = manhattan------
    l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 5, 'metric':'manhattan'}, 0)  #'eom', 'leaf' 'euclidean' 'metric':'manhattan'   , 'metric':'pyfunc', 'func':cu.cosine_metric
    # assess clustering
    # сначала надо переименовать мусорный -1 кластер в последний какой то
    l_cluster_cnt = l_labes_hdbscan.max()
    print("HDBScan has found N clusters (without noise one)>> ", l_cluster_cnt)
    l_labes_hdbscan = [l_cluster_cnt if i == -1 else i for i in l_labes_hdbscan]
    # print(l_labes_hdbscan[0:100])
    # l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, -1)
    # print("Clustering compare avg precision and accuracy for dbscan", l_assess_res)

    # time.sleep(1.1)  # pause 1.1 seconds  это нужно чтобы логи не перетерлись
    l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, l_cluster_cnt)
    l_garbage_cluster_size = len([i for i in l_labes_hdbscan if i == l_cluster_cnt])
    l_assess_res.append(l_garbage_cluster_size)
    l_assess_res.append(l_cluster_cnt)
    # print("Clustering compare avg precision and accuracy for dbscan without garbage cluster ( indicatores removed) "
    #       , l_garbage_cluster_size
    #       , " indicatores removed) "
    #       , l_assess_res)
    l_alg_list.append('HDBScan min-size=5 distance = manhattan')
    l_assess_res_list.append(l_assess_res)

    if p_performe_cosine_dist == 1:

        # --- HDBScan min=17 distance = cosine------
        l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 17, 'metric':'pyfunc', 'func':cu.cosine_metric}, 0)  #'eom', 'leaf' 'euclidean' 'metric':'manhattan'   , 'metric':'pyfunc', 'func':cu.cosine_metric
        # assess clustering
        # сначала надо переименовать мусорный -1 кластер в последний какой то
        l_cluster_cnt = l_labes_hdbscan.max()
        print("HDBScan has found N clusters (without noise one)>> ", l_cluster_cnt)
        l_labes_hdbscan = [l_cluster_cnt if i == -1 else i for i in l_labes_hdbscan]
        # print(l_labes_hdbscan[0:100])
        # l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, -1)
        # print("Clustering compare avg precision and accuracy for dbscan", l_assess_res)

        # time.sleep(1.1)  # pause 1.1 seconds  это нужно чтобы логи не перетерлись
        l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, l_cluster_cnt)
        l_garbage_cluster_size = len([i for i in l_labes_hdbscan if i == l_cluster_cnt])
        l_assess_res.append(l_garbage_cluster_size)
        l_assess_res.append(l_cluster_cnt)
        # print("Clustering compare avg precision and accuracy for dbscan without garbage cluster ( indicatores removed) "
        #       , l_garbage_cluster_size
        #       , " indicatores removed) "
        #       , l_assess_res)
        l_alg_list.append('HDBScan min-size=17 distance = cosine')
        l_assess_res_list.append(l_assess_res)

        # --- HDBScan min=7 distance = cosine------
        l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 7, 'metric':'pyfunc', 'func':cu.cosine_metric}, 0)  #'eom', 'leaf' 'euclidean' 'metric':'manhattan'   , 'metric':'pyfunc', 'func':cu.cosine_metric
        # assess clustering
        # сначала надо переименовать мусорный -1 кластер в последний какой то
        l_cluster_cnt = l_labes_hdbscan.max()
        print("HDBScan has found N clusters (without noise one)>> ", l_cluster_cnt)
        l_labes_hdbscan = [l_cluster_cnt if i == -1 else i for i in l_labes_hdbscan]
        # print(l_labes_hdbscan[0:100])
        # l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, -1)
        # print("Clustering compare avg precision and accuracy for dbscan", l_assess_res)

        # time.sleep(1.1)  # pause 1.1 seconds  это нужно чтобы логи не перетерлись
        l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, l_cluster_cnt)
        l_garbage_cluster_size = len([i for i in l_labes_hdbscan if i == l_cluster_cnt])
        l_assess_res.append(l_garbage_cluster_size)
        l_assess_res.append(l_cluster_cnt)
        # print("Clustering compare avg precision and accuracy for dbscan without garbage cluster ( indicatores removed) "
        #       , l_garbage_cluster_size
        #       , " indicatores removed) "
        #       , l_assess_res)
        l_alg_list.append('HDBScan min-size=7 distance = cosine')
        l_assess_res_list.append(l_assess_res)

        # --- HDBScan min=5 distance = cosine------
        l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 5, 'metric':'pyfunc', 'func':cu.cosine_metric}, 0)  #'eom', 'leaf' 'euclidean' 'metric':'manhattan'   , 'metric':'pyfunc', 'func':cu.cosine_metric
        # assess clustering
        # сначала надо переименовать мусорный -1 кластер в последний какой то
        l_cluster_cnt = l_labes_hdbscan.max()
        print("HDBScan has found N clusters (without noise one)>> ", l_cluster_cnt)
        l_labes_hdbscan = [l_cluster_cnt if i == -1 else i for i in l_labes_hdbscan]
        # print(l_labes_hdbscan[0:100])
        # l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, -1)
        # print("Clustering compare avg precision and accuracy for dbscan", l_assess_res)

        # time.sleep(1.1)  # pause 1.1 seconds  это нужно чтобы логи не перетерлись
        l_assess_res = ca.getClusteringQuality(l_labes_hdbscan, p_ind_target, l_cluster_cnt)
        l_garbage_cluster_size = len([i for i in l_labes_hdbscan if i == l_cluster_cnt])
        l_assess_res.append(l_garbage_cluster_size)
        l_assess_res.append(l_cluster_cnt)
        # print("Clustering compare avg precision and accuracy for dbscan without garbage cluster ( indicatores removed) "
        #       , l_garbage_cluster_size
        #       , " indicatores removed) "
        #       , l_assess_res)
        l_alg_list.append('HDBScan min-size=5 distance = cosine')
        l_assess_res_list.append(l_assess_res)



    #избавимся от искуственно добавленной фичи с нулями чтобы избежать ненужной нормироваки
    # l_labes_hdbscan = [i for i in ranage(len(l_labes_hdbscan)-1)]
    # l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 30}, 0)  #'eom', 'leaf'
    # l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 21}, 0)  #'eom', 'leaf'
    # l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 17}, 0)  #'eom', 'leaf'
    # l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 7}, 0)  #'eom', 'leaf'
    # l_labes_hdbscan = doClustering(p_features, hdbscan.HDBSCAN, (), {'min_cluster_size': 7, 'min_samples': 1, 'cluster_selection_method': 'leaf'}, 0)  # 'eom', 'leaf'

    # print("\n l_labes_kmeans >> ", l_labes_kmeans[0:100])
    # print("\n l_ind_target >> ", l_ind_target[0:100])

    # print("\n count l_ind_target >> ", len(p_ind_target))
    # print("\n count l_labes_hdbscan >> ", len(l_labes_hdbscan))


    return l_alg_list, l_assess_res_list


#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------


def main():
  print("Start")
  print("--- Start time %s ---" % strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
  start_time = time.time()

  # Настраиваем параметры эксперимента
  l_exp_conf = exp_conf.getExperimentConfigByName('edu_gen')
  path = l_exp_conf['path']
  prefix = l_exp_conf['prefix']
  suffix = l_exp_conf['suffix']
  start_th = l_exp_conf['start_th']  # start threshold for fuzzy graph

  hier_graph_file = path + prefix + "words_hier_graph_" + suffix + "_th_" + str(round(start_th * 100)) + ".graphml"
  clustering_results_file = path + prefix + "hier_" + suffix + "_th_" + str(round(start_th * 100))
  # crisp_hier_clustering_results_file = path + prefix + suffix + "_th_" + str(round(start_th * 100)) #+ "_hdbscan"
  l_target_file = l_exp_conf['target_file']
  l_target_list = l_exp_conf['target_list']

  # data = [[1,1,1,1],[0,1,1,1],[0,0,1,1],[0,-1,1,1],[-1,-1,1,1]]


  [l_ind_names, l_ind_target, l_features_names, l_features_list] = getClusteringTask(hier_graph_file, clustering_results_file, l_target_file, l_target_list)

  # print("\n l_ind_names >> ", l_ind_names[0:10])
  print("\n count of l_ind_names >> ", len(l_ind_names))
  print("\n count of l_ind_target >> ", len(l_ind_target))


  # Перебираем все полученные фичи, для каждой выполняем все алгоритмы и результаты сводим в одну таблицу и записваем в лог
  l_cluster_res_list = []
  for fn in range(len(l_features_names)):
      l_is_cos = 0
      if 'word2vec' in l_features_names[fn]:
          l_is_cos = 1
      try:
          [l_alg_list, l_assess_res_list] = performClusteringForFetures(l_features_list[fn], l_ind_target, l_is_cos)
          for alg_n in range(len(l_alg_list)):
              l_res = [l_features_names[fn], l_alg_list[alg_n]]
              for l_mark in l_assess_res_list[alg_n]:
                  l_res.append(l_mark)
              l_cluster_res_list.append( l_res )
      except Exception:
          print("Error during clustering feature", l_features_names[fn])

  cu.writeMatrixToLog("Experiment_result_matrix", l_cluster_res_list, l_exp_conf["path"])


  #--------------    FLAT FEATURES -----------------------------
  # print("\n --------------   Flat features   ----------------------")
  # l_assess_flat = performClusteringForFetures(l_features_flat, l_ind_target)


  #--------------    WEIGHTED FEATURES -----------------------------
  # print("\n --------------   Weighted features   ----------------------")
  # l_assess_res = performClusteringForFetures(l_features_lognorm, l_ind_target)


  #--------------    Word2Vec FEATURES -----------------------------
  # print("\n --------------   Word2Vec features   ----------------------")
  # l_assess_res = performClusteringForFetures(l_features_w2v, l_ind_target)


  #--------------    WEIGHTED FEATURES LIST (mu in [2, 11.9]) -----------------------------
  # l_assess_list = [l_assess_flat]
  # # l_assess_list.append(l_assess_flat)
  # for f in features_list:
  #   l_assess_res = performClusteringForFetures(f, l_ind_target)
  #   l_assess_list.append(l_assess_res)
  #
  # cu.writeMatrixToLog("LogNormClusteringEperiment_1_6", l_assess_list)

  # plt.scatter(T[0], T[1], c='b', **plot_kwds)
  # frame = plt.gca()
  # frame.axes.get_xaxis().set_visible(False)
  # frame.axes.get_yaxis().set_visible(False)
  # plt.show()

  # path = "edu_inds_clustering/"
  # # max_inds_count = 4005
  # prefix = "171003_inds_"
  # suffix = "pre"
  # start_th = 0.3 # start threshold for fuzzy graph
  #
  #
  # hier_graph_file = path + prefix + "words_hier_graph_" + suffix + "_th_" + str(round(start_th * 100)) + ".graphml"
  # clustering_results_file = path + prefix + "hier_" + suffix + "_th_" + str(round(start_th * 100))
  # crisp_hier_clustering_results_file = path + prefix + suffix + "_th_" + str(round(start_th * 100)) #+ "_hdbscan"
  #
  # clusterIndicators(hier_graph_file, clustering_results_file, crisp_hier_clustering_results_file)


  print("Done")
  end_time = time.time()
  print("--- %s seconds ---" % (end_time - start_time))


#-----------------------------------------------------------------------------------------------------------------------

def testLogNormal():
  l_y_list = []
  for k in range(50):
      y = showLogNormal(1+k/10, 0)
      l_y_list.append(y)
  cu.writeMatrixToLog("LogNormFunction_1_6", l_y_list)


# -----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # w2v vectors  - загрузим один раз, т.к. это дорогая операция
    vectors_file = "data/" + "ruwikiruscorpora_0_300_20.bin"
    model = KeyedVectors.load_word2vec_format(vectors_file, binary=True)
    model.init_sims(True)
    print("Vectors loaded")
    # w = "учитель_NOUN"
    # if w in model.vocab:
    #     l_wv = model.word_vec(w, use_norm = True)
    # print(l_wv)
    main()
    # testLogNormal()
