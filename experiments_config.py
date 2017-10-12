


#-----------------------------------------------------------------------------------------------------------------------


def getExperimentConfigByName(p_experiment_name):
    """

    :param p_experiment_name:
     edu_pre - дошкольное образование выполнялось 2017.10.03 (для статьи на ФТИ-17)
     edu_gen - общее образование, эксперимент для проверки фич (логнормальных и w2v с косинусом и без)
    :return: словарь с параметрами эксперимента

    """
    l_dic = {}
    if p_experiment_name == 'edu_pre':
        l_dic['path'] =  "edu_inds_clustering/pre/"
        l_dic['max_inds_count'] = 4005
        l_dic['prefix'] = "171003_inds_"
        l_dic['suffix'] = "pre"
        l_dic['start_th'] = 0.3  # start threshold for fuzzy graph
        l_dic['in_file'] =  l_dic.get('path') + "дошкольное_образование_все.xlsx"
        l_dic['target_file'] =  l_dic.get('path') + "edu_classif_new_3.xlsx"
        l_dic['target_list'] = 'pre_revised'

    l_dic = {}
    if p_experiment_name == 'edu_pre_20171012':
        l_dic['path'] =  "edu_inds_clustering/20171012_pre/"
        l_dic['max_inds_count'] = 4005
        l_dic['prefix'] = "inds_"
        l_dic['suffix'] = "pre"
        l_dic['start_th'] = 0.3  # start threshold for fuzzy graph
        l_dic['in_file'] =  l_dic.get('path') + "дошкольное_образование_все.xlsx"
        l_dic['target_file'] =  l_dic.get('path') + "кластеризация_образование_группировка.xlsx"
        l_dic['target_list'] = 'pre_revised'

    if p_experiment_name == 'edu_gen':
        l_dic['path'] =  "edu_inds_clustering/20171010_gen/"
        l_dic['max_inds_count'] = 5633
        l_dic['prefix'] = "inds_"
        l_dic['suffix'] = "gen"
        l_dic['start_th'] = 0.3  # start threshold for fuzzy graph
        l_dic['in_file'] = l_dic.get('path') + "общее_образование.xlsx"
        l_dic['target_file'] =  l_dic.get('path') + "edu_classif_new_3.xlsx"
        l_dic['target_list'] = 'gen_revised'

    return l_dic

#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------



def main():

    res = getExperimentConfigByName('edu_gen')
    print(res)




#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
