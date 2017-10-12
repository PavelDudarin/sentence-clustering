import pandas as pd
import datetime
import numpy as np
import scipy
import sklearn.metrics.pairwise as pw

#-----------------------------------------------------------------------------------------------------------------------

def writeMatrixToLog(p_file_suffix, p_array, p_path = None):

    # write matrix  to csv log file
    df = pd.DataFrame(p_array)
    now = datetime.datetime.now()
    l_prefix = now.strftime("%Y%m%d_%H%M%S") + "_"
    l_path = p_path
    if l_path is None:
        l_path = 'log/'
    df.to_csv(l_path+ l_prefix + p_file_suffix + "_log.csv", index=False, header=True, encoding="utf-8")
    print(p_file_suffix + " log saved in ", l_path + l_prefix + p_file_suffix + "_log.csv")

#-----------------------------------------------------------------------------------------------------------------------

def makeAverageVector(vectors, weights = None):
    n_vectors = len(vectors)
    if not weights:
        weights = np.ones(n_vectors)
    avg_vector = np.multiply(vectors[0], weights[0])
    if n_vectors > 1:
        for i in range(1,n_vectors):
            avg_vector = np.add(avg_vector, np.multiply(vectors[i], weights[i]))
    return np.divide(avg_vector, n_vectors)

#-----------------------------------------------------------------------------------------------------------------------

def getIdf(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

#-----------------------------------------------------------------------------------------------------------------------

def cosine_metric(v1, v2):
    """
    возвращает дистанцию на основе косинуса
    """
    # return scipy.spatial.distance.cosine(v1,v2)
    return pw.cosine_similarity(v1,v2)[0][0]
    # return 0.5

#-----------------------------------------------------------------------------------------------------------------------

# def vec_sum_of_lists(l1, l2):
#     l_sum = [l1[i]+l2[i] for i in range(len(l1))]
#     return l_sum

#-----------------------------------------------------------------------------------------------------------------------

def main():

    l1 = [10, 20, 30]
    l2 = [1,2,3]
    # s = sum(l1, l2)
    # print(s)
    sim = pw.cosine_similarity(l1, l2)
    print(sim)
    dist = pw.cosine_distances(l1, l2)
    print(dist[0][0])


if __name__ == "__main__":
    main()
