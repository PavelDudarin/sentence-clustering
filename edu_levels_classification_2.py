import pickle
import numpy as np
import pandas as pd
import nltk

path = "edu_classif/"
ind_file = path+"monocity_inds_105_all_edu.pkl"

pkl_file = open(ind_file, "rb")
inds = pickle.load(pkl_file)
pkl_file.close()

levels_keywords = { 
    "general" : [
        ["общеобразовательный_ADJ", "егэ_NOUN", "школьник_NOUN", "учитель_NOUN", "школа_NOUN", 
         "гимназия_NOUN", "лицей_NOUN"],
        ["государственный_ADJ экзамен_NOUN", "среднее_NOUN образование_NOUN", "общий_ADJ образование_NOUN"]
        ],
    
    "higher" : [
        ["вуз_NOUN", "институт_NOUN", "студент_NOUN", "университет_NOUN",
        "стипендиат_NOUN", "стипендия_NOUN", "бакалавриат_NOUN", "диплом_NOUN", "профессорско_ADJ", 
        "профессор_NOUN", "магистратура_NOUN", "магистрант_NOUN", "бакалавр_NOUN"],
        ["высокий_ADJ образование_NOUN", "высокий_ADJ профессиональный_ADJ", "выпуск_NOUN квалификация_NOUN"]
        ],
    
    "addit" : [
        ["кружок_NOUN", "секция_NOUN"],
        ["дополнительный_ADJ образование_NOUN", "школа_NOUN искусство_NOUN", "музыкальный_ADJ школа_NOUN"]
        ],
    
    "qual" : [
        ["квалификация_NOUN", "переподготовка_NOUN", "тренинг_NOUN"],
        []
        ],
    
    "pre" : [
        ["дошкольный_ADJ", "доу_NOUN", "предшкольный_ADJ", "дошкольник_NOUN"],
        ["детский_ADJ сад_NOUN"]
        ]}

levels = {"qual":[], "higher":[], "pre":[], "general":[], "addit":[]}

levels_exclude = {"qual":[[],[]], 
                  "higher":[
                      ["общеобразовательный_ADJ", "дошкольный_ADJ", 
                       "доу_NOUN", "школа_NOUN"], 
                      ["дополнительный_ADJ образование_NOUN", "институт_NOUN семья_NOUN"]
                      ],
                  "pre":[
                      ["общеобразовательный_ADJ", "вуз_NOUN", "студент_NOUN", 
                       "университет_NOUN"], 
                      ["дополнительный_ADJ образование_NOUN"]
                      ],
                  "general":[
                      ["вуз_NOUN", "студент_NOUN", "университет_NOUN", 
                       "дошкольный_ADJ", "доу_NOUN"],
                      ["дополнительный_ADJ образование_NOUN"]
                      ],
                  "addit":[
                      ["общеобразовательный_ADJ", "вуз_NOUN", "университет_NOUN", "дошкольный_ADJ", 
                       "доу_NOUN"],
                      []
                      ]}

overlap_inds = []

for ind in inds:
    if len(ind[2]) > 0:
        bigrm = list(nltk.bigrams(ind[2]))
        bigrm = [" ".join(b) for b in bigrm]
        for level in levels:
            if ( (any(word in ind[2] for word in levels_keywords[level][0])
                  or any(b in bigrm for b in levels_keywords[level][1]) )
                     and not (any(word in ind[2] for word in levels_exclude[level][0])
                          or any(b in bigrm for b in levels_exclude[level][1]))
                ):
                levels[level].append(ind)
            elif ( (any(word in ind[2] for word in levels_keywords[level][0])
                  or any(b in bigrm for b in levels_keywords[level][1]) )
                    and (any(word in ind[2] for word in levels_exclude[level][0])
                        or any(b in bigrm for b in levels_exclude[level][1])
                            and "институт_NOUN семья_NOUN" not in bigrm)
                ):
                if ind not in overlap_inds:
                    overlap_inds.append(ind)

for level in levels:
    output = open(path+"edu_inds_level_"+level+".pkl", "wb")
    pickle.dump(levels[level], output)
    output.close()
    
    df = pd.DataFrame(levels[level])
    df.to_csv(path+"edu_inds_level_"+level+".csv", index=False, header=True, encoding = "utf-8")    
    
    print(level, len(levels[level]))

df = pd.DataFrame(overlap_inds)
df.to_csv(path+"edu_inds_level_overlap.csv", index=False, header=True, encoding = "utf-8")    
print("overlap", len(overlap_inds))
