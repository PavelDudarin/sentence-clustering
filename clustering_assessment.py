import numpy as np

#--------------------------------------------------------------------------

def getClusteringQuality(clusteringToAssess, clusteringToCompare):

    l_rows = max(clusteringToAssess[1])+1
    l_cols = max(clusteringToCompare[1])+1
    l_compare = [[0 for x in range(l_cols+2)] for y in range(l_rows)]
    # print(l_rows, l_cols, l_compare)
    # отсортируем массивы чтобы потом поиск был быстрее
    # clusteringToAssess = sorted(clusteringToAssess)
    # clusteringToCompare = sorted(clusteringToCompare)

    for i in range(len(clusteringToAssess[0])):
        l_compare[clusteringToAssess[1][i]][clusteringToCompare[1][i]] += 1
    # print(l_rows, l_cols, l_compare)

    # assessment
    for i in range(l_rows):
        l_compare[i][l_cols] = max(l_compare[i]) / sum(l_compare[i]) # precision  - сколько попало в цель ?
        l_compare[i][l_cols+1] = max(l_compare[i]) / sum( [l_compare[j][l_compare[i].index(max(l_compare[i]))] for j in range(l_rows) ]) # accuracy - насколько чист полученный кластер
        # print(l_compare[i].index(max(l_compare[i])))
        # print([l_compare[0:1][0]])
        # print(l_compare[:][l_compare[i].index(max(l_compare[i]))])

    l_precision_avg = np.average([l_compare[j][l_cols] for j in range(l_rows)])
    l_accuracy_avg = np.average([l_compare[j][l_cols+1] for j in range(l_rows)])

    return [l_precision_avg, l_accuracy_avg]

#--------------------------------------------------------------------------

def main():

    l_clustering1 = [['a', 'b', 'c', 'd', 'e', 'f'], [0, 1, 1, 1, 0, 1]]
    l_clustering2 = [['a', 'b', 'c', 'd', 'e', 'f'], [0, 1, 1, 1, 0, 1]]
    l_clustering3 = [['a', 'b', 'c', 'd', 'e', 'f'], [1, 1, 1, 1, 1, 1]]
    l_clustering4 = [['a', 'b', 'c', 'd', 'e', 'f'], [0, 1, 2, 3, 4, 5]]

    l_res = getClusteringQuality(l_clustering1, l_clustering3)
    print("Clustering compare matrix ", l_res)


if __name__ == "__main__":
    main()
