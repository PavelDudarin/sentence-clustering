import numpy as np
# import pandas as pd
import common_utils as cu
# import math as m

#-----------------------------------------------------------------------------------------------------------------------

# вычисляет точность и аккуратность (складывает в последний и предпоследний столбцы)
def calcPrecisionAndAccuracy(p_compare, p_rows, p_cols):
    # assessment rows (precision and accuracy)
    for i in range(p_rows):
        l_row_sum = sum(p_compare[i])
        # такое может быть только в случае если мы удалили мусорный кластер, поэтому логично пометить его особым образом
        if l_row_sum == 0:
            p_compare[i][p_cols] = 0
            p_compare[i][p_cols+1] = 0
        else:
            p_compare[i][p_cols] = max(p_compare[i]) / sum(p_compare[i]) # precision  - сколько попало в цель ?
            l_col_index = p_compare[i].index(max(p_compare[i]))
            l_col_sum = sum([p_compare[j][l_col_index] for j in range(p_rows)])  # accuracy - насколько чист полученный кластер
            if l_col_sum > 0:
                p_compare[i][p_cols + 1] = max(p_compare[i]) / l_col_sum
            else:
                p_compare[i][p_cols + 1] = 0
        # print(l_compare[i].index(max(l_compare[i])))
        # print([l_compare[0:1][0]])
        # print(l_compare[:][l_compare[i].index(max(l_compare[i]))])


#-----------------------------------------------------------------------------------------------------------------------

# вычисляет чистоту (складывает в последний ряд)
def calcPurity(p_compare, p_rows, p_cols):
    # assessment cols (precision and accuracy)
    for i in range(p_cols):
        l_col = [p_compare[ri][i] for ri in range(p_rows)]
        l_col_sum = sum(l_col)
        # у нас могут быть пустые кластеры, и их мы пометим отдельно, чтобы вернуть два варианта оценки
        if l_col_sum == 0:
            p_compare[p_rows][i] = 0
        else:
            p_compare[p_rows][i] = max(l_col) / l_col_sum # purity

#-----------------------------------------------------------------------------------------------------------------------

def calc_TP_FN_FP_TN_Total(p_compare, p_rows, p_cols):

  #Total
  l_total = sum([ sum(p_compare[ri][ci] for ci in range(p_cols)) for ri in range(p_rows)])
  l_total = int(round(l_total*(l_total-1)/2))

  #TP
  l_tp = sum([ sum(int(round((p_compare[ri][ci]*(p_compare[ri][ci]-1))/2)) for ci in range(p_cols)) for ri in range(p_rows)])

  #TN
  l_tn = int(round(
          sum([ sum(
                    p_compare[ri][ci] * sum([ sum(p_compare[ri1][ci1] for ci1 in range(p_cols) if ci1 != ci) for ri1 in range(p_rows) if ri1 != ri])

                    for ci in range(p_cols)
                   )
               for ri in range(p_rows)
             ]) / 2))

  #FP
  l_fp = int(round(
          sum([ sum(
                     p_compare[ri][ci] * sum(p_compare[ri1][ci] for ri1 in range(p_rows) if ri1 != ri)

                    for ci in range(p_cols)
                   )
               for ri in range(p_rows)
             ]) / 2))

  #FN
  l_fn = int(round(
          sum([ sum(
                     p_compare[ri][ci] * sum(p_compare[ri][ci1] for ci1 in range(p_cols) if ci1 != ci)

                    for ci in range(p_cols)
                   )
               for ri in range(p_rows)
             ]) / 2))

  #check
  if l_tp + l_tn + l_fp + l_fn != l_total:
      print("Error in calc_TP_FN_FP_TN_Total the equation l_tp + l_tn + l_fp + l_fn == l_total is not correct", l_tp + l_tn + l_fp + l_fn , l_total)

  return [l_tp, l_tn, l_fp, l_fn, l_total]

#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------


#Получение оценок [точность, аккуратность] для кластериации
# вначале передается список с эталонной кластеризацией
# затем передается список с кластеризацией для оценки
# список в формате [[item1, item2, item3...],[cluter_n_0, cluter_n_1, cluter_n_2, ...]]
# совпадение количества кластеров не требуется, но количество item (элементов) должно совпадать
# garbadge_cluster_n - аналог шумового кластера, с той разницей что в эталонной кластеризации шума нет подразумевается
# если garbadge_cluster_n отсутсвует то передаем -1
# результатом будут два варианта точности и аккуратности, с разной трактовкой пустых строк/стобцов. Сначала трактуем как 0, потом не учитываем
def getClusteringQuality(clusteringToAssess, clusteringToCompare, garbage_cluster_n = -1):
    print("Start getClusteringQuality")
    l_rows = max(clusteringToCompare)+1
    l_cols = max(clusteringToAssess)+1
    # l_compare = [[0 for x in range(l_cols + 2)] for y in range(l_rows)]
    l_compare = [[0 for x in range(l_cols + 2)] for y in range(l_rows + 1)]
    # print(l_rows, l_cols, l_compare)
    # отсортируем массивы чтобы потом поиск был быстрее
    # clusteringToAssess = sorted(clusteringToAssess)
    # clusteringToCompare = sorted(clusteringToCompare)

    for i in range(len(clusteringToCompare)):
        l_compare[clusteringToCompare[i]][clusteringToAssess[i]] += 1
    # print(l_rows, l_cols, l_compare)

    #удалим мусотрный кластер если он есть
    if garbage_cluster_n > -1:
        l_compare = [  [row[col]  for col in range(len(row)) if col != garbage_cluster_n]  for row in l_compare]
        l_cols -= 1

    calcPrecisionAndAccuracy(l_compare, l_rows, l_cols)
    calcPurity(l_compare, l_rows, l_cols)

    # assessment rows (precision and accuracy)
    # for i in range(l_rows):
    #     l_row_sum = sum(l_compare[i])
    #     # такое может быть только в случае если мы удалили мусорный кластер, поэтому логично пометить его особым образом
    #     if l_row_sum == 0:
    #         l_compare[i][l_cols] = 0
    #         l_compare[i][l_cols+1] = 0
    #     else:
    #         l_compare[i][l_cols] = max(l_compare[i]) / sum(l_compare[i]) # precision  - сколько попало в цель ?
    #         l_col_index = l_compare[i].index(max(l_compare[i]))
    #         l_col_sum = sum([l_compare[j][l_col_index] for j in range(l_rows)])  # accuracy - насколько чист полученный кластер
    #         if l_col_sum > 0:
    #             l_compare[i][l_cols + 1] = max(l_compare[i]) / l_col_sum
    #         else:
    #             l_compare[i][l_cols + 1] = 0
    #     # print(l_compare[i].index(max(l_compare[i])))
    #     # print([l_compare[0:1][0]])
    #     # print(l_compare[:][l_compare[i].index(max(l_compare[i]))])

    # assessment cols (precision and accuracy)
    # for i in range(l_cols):
    #     l_col = [l_compare[ri][i] for ri in range(l_rows)]
    #     l_col_sum = sum(l_col)
    #     # у нас могут быть пустые кластеры, и их мы пометим отдельно, чтобы вернуть два варианта оценки
    #     if l_col_sum == 0:
    #         l_compare[l_rows][i] = 0
    #         l_compare[l_rows+1][i] = 0
    #     else:
    #         l_compare[l_rows][i] = max(l_col) / l_col_sum # precision  - сколько попало в цель ?
    #         # accuracy - насколько чист полученный кластер
    #         l_row_index = l_col.index(max(l_col))
    #         l_row_sum = sum(l_compare[l_row_index][c] for c in range(l_cols))
    #         if l_row_sum > 0:
    #             l_compare[l_rows+1][i] = max(l_col) / l_row_sum
    #         else:
    #             l_compare[l_rows+1][i] = 0

        # print(l_compare[i].index(max(l_compare[i])))
        # print([l_compare[0:1][0]])
        # print(l_compare[:][l_compare[i].index(max(l_compare[i]))])

    # assessment -- OLD VERSION
    # for i in range(l_rows):
    #     l_row_sum = sum(l_compare[i])
    #     if l_row_sum == 0:
    #         l_compare[i][l_cols] = 0
    #         l_compare[i][l_cols + 1] = 0
    #     else:
    #         l_compare[i][l_cols] = max(l_compare[i]) / sum(l_compare[i]) # precision  - сколько попало в цель ?
    #         l_compare[i][l_cols+1] = max(l_compare[i]) / sum( [l_compare[j][l_compare[i].index(max(l_compare[i]))] for j in range(l_rows) ]) # accuracy - насколько чист полученный кластер
    #     # print(l_compare[i].index(max(l_compare[i])))
    #     # print([l_compare[0:1][0]])
    #     # print(l_compare[:][l_compare[i].index(max(l_compare[i]))])

    # write matrix  to csv log file
    cu.writeMatrixToLog("getClusteringQuality", l_compare)
    # df = pd.DataFrame(l_compare)
    # now = datetime.datetime.now()
    # l_prefix = now.strftime("%Y%m%d_%H%M%S")+"_"
    # df.to_csv("log/" + l_prefix + "getClusteringQuality_log.csv", index=False, header=True, encoding="utf-8")
    # print("getClusteringQuality log results saved in ", "log/" + l_prefix + "getClusteringQuality_log.csv")



    l_precision_avg_0 = np.average([l_compare[j][l_cols] for j in range(l_rows)])
    l_precision_avg_1 = np.average([l_compare[j][l_cols] for j in range(l_rows) if l_compare[j][l_cols] > 0])

    l_accuracy_avg_0 = np.average([l_compare[j][l_cols+1] for j in range(l_rows)])
    l_accuracy_avg_1 = np.average([l_compare[j][l_cols+1] for j in range(l_rows) if l_compare[j][l_cols+1] > 0])

    l_purity_avg_0 = np.average([l_compare[l_rows][j] for j in range(l_cols)])
    l_purity_avg_1 = np.average([l_compare[l_rows][j] for j in range(l_cols) if l_compare[l_rows][j] > 0])

    # l_col_accuracy_avg_0 = np.average([l_compare[l_rows+1][j] for j in range(l_cols)])
    # l_col_accuracy_avg_1 = np.average([l_compare[l_rows+1][j] for j in range(l_cols) if l_compare[l_rows+1][j] > 0])

    [l_tp,l_tn,l_fp,l_fn,l_total] = calc_TP_FN_FP_TN_Total(l_compare, l_rows, l_cols)

    print("Finish getClusteringQuality")
    l_ri = 0
    l_precision = 0
    l_recall = 0
    l_f1_score = 0
    try:
      l_ri =  (l_tp+l_tn)/(l_total)
      l_precision = (l_tp)/(l_tp+l_fp)
      l_recall =  (l_tp)/(l_tp+l_fn)
      l_f1_score = (2*l_precision*l_recall) / (l_precision+l_recall)
    except Exception:
        pass

    # print(l_compare)
    return [l_precision_avg_0, l_accuracy_avg_0, l_purity_avg_0, #l_col_accuracy_avg_0,
            l_precision_avg_1, l_accuracy_avg_1, l_purity_avg_1, (l_precision_avg_1+l_accuracy_avg_1+l_purity_avg_1)/3,  #l_col_accuracy_avg_1,
            l_tp, l_tn, l_fp, l_fn, l_total,
            # RI                     Pricesion                Recall             F1-score
            l_ri, l_precision, l_recall, l_f1_score
           ]

#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------

def main():

    l_clustering1 = [0, 1, 1, 1, 0, 1]
    l_clustering2 = [0, 1, 1, 1, 0, 1]
    l_clustering3 = [1, 1, 1, 1, 1, 1]
    l_clustering4 = [0, 1, 2, 3, 4, 5]

    l_res = getClusteringQuality(l_clustering4, l_clustering1)
    print("Clustering compare matrix ", l_res)



if __name__ == "__main__":
    main()
