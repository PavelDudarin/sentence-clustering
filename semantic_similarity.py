from gensim.models.keyedvectors import KeyedVectors
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------

def getSemantiSimilarity(vectors_file, word_pairs):
    print("Start getSemantiSimilarity")

    words_ds = {wp[0]+"_"+wp[1]:0  for wp in word_pairs}

    # w2v vectors
    model = KeyedVectors.load_word2vec_format(vectors_file, binary=True)
    print("Vectors loaded")

    for wp0, wp1 in word_pairs:
        sim = 0
        if wp0 in model.vocab and wp1 in model.vocab:
            sim = model.similarity(wp0, wp1)
        words_ds[wp0+"_"+wp1] = sim

    print("getSemantiSimilarity finished")
    return words_ds

#-----------------------------------------------------------------------------------------------------------------------


def main():
  print("Start")
  vectors_file = "data/" + "ruwikiruscorpora_0_300_20.bin"

  l_word_pairs = [["работник_NOUN", "работник_NOUN"],["рабочий_ADJ", "работник_NOUN"],["рабочий_NOUN", "работник_NOUN"],["работодатель_NOUN", "работник_NOUN"],["работник_NOUN", "персонал_NOUN"],["персонал_NOUN", "работник_NOUN"]]
  # вывод нет смысла брать пары связанные меньше чем на 0.5 там уже мусор начинается

  l_res = getSemantiSimilarity(vectors_file, l_word_pairs)
  print(l_res)

  print("Done")


#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()