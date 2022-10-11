import re, os, csv, json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def get_rank_score(cls_prob, y, sklaern_cls_id_map=None, dise2id=None):
    assert len(cls_prob) == len(y)

    mrr_list, hits_1_cnt, hits_2_cnt = [], 0, 0
    for i in range(len(cls_prob)):
        goals_dict = {id:prob for id,prob in enumerate(cls_prob[i])}

        if sklaern_cls_id_map:
            goals_dict = {int(dise2id[sklaern_cls_id_map[id]]):prob for id,prob in enumerate(cls_prob[i])}

        target = goals_dict[y[i]]

        less_cnt = 0
        for k,v in goals_dict.items():
            if k != y[i] and v < target:
                less_cnt += 1
        rank = len(goals_dict) - less_cnt # worst case 
        assert rank >= 1 and rank <= len(goals_dict)

        if rank == 1:
            hits_1, hits_2 = 1, 1
        elif rank == 2:
            hits_1, hits_2 = 0, 1
        else:
            hits_1, hits_2 = 0, 0

        mrr_list.append(1.0/rank)
        hits_1_cnt += hits_1
        hits_2_cnt += hits_2

    return np.mean(mrr_list), 1.0*hits_1_cnt/len(y), 1.0*hits_2_cnt/len(y)


def load_KG(KG_data_path):
    ent2id, id2ent = {}, {}
    with open(KG_data_path + 'entity2id.txt', 'r', encoding='utf-8') as f:
        for idx, item in enumerate(f.readlines()):
            if idx != 0:
                ent, id = item.split('\t')
                ent, id = ent.strip('\n'), int(id.strip('\n'))
                ent2id[ent] = id
                id2ent[id] = ent
 
    rel2id, id2rel = {}, {}
    with open(KG_data_path + 'relation2id.txt', 'r', encoding='utf-8') as f:
        for idx, item in enumerate(f.readlines()):
            if idx != 0:
                rel, id = item.split('\t')
                rel, id = rel.strip('\n'), int(id.strip('\n'))
                rel2id[rel] = int(id)
                id2rel[id] = rel

    embedings = pickle.load(open(KG_data_path + 'TransE.pkl', 'rb'))

    # embedings = {'zero_const':'...', 'pi_const':'...', 'ent_embeddings.weight':'...', 'rel_embeddings.weight':'...'}

    print(f'KG loaded with {len(ent2id)} entities,  {len(rel2id)} relations')
    return ent2id, id2ent, rel2id, id2rel, embedings

def load_rules(rule_path,  K_fold=None, incre_dise=None):
    rule_dict = {}
    cnt = 0
    if K_fold is not None:
        rule_path += f'K-fold/fold-{K_fold}'

    for file in os.listdir(rule_path):
        csv_file = os.path.join(rule_path, file)
        if '.csv' in csv_file:
            if incre_dise is not None and incre_dise in csv_file:
                continue
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for item in reader:
                    if reader.line_num == 1:
                        continue
                    symtoms = re.findall(re.compile(r'[(](.*?)[)]',re.S), item[0])[0].split(',')
                    symtoms = [i.strip().strip("'") for i in symtoms]
                    # print(symtoms)
                    disease = re.findall('(?<=THEN).*$', item[0])[0].strip()
                    # print(disease)
                    rule_dict[f'r{cnt}'] = (symtoms, [disease], float(item[1]))
                    cnt += 1
    print(f'{len(rule_dict)} rules are added!')
    return rule_dict

def load_data(data_path, split):
    data = []  
    if split == 'train + valid':
        diag_file = [data_path + 'diagnose_train.json', data_path + 'diagnose_valid.json']
    else:
        diag_file = data_path + f'diagnose_{split}.json'
    
    if split != 'train + valid':
        with open(diag_file, 'r', encoding='utf-8') as f1:
            for line in f1.readlines():
                data.append(json.loads(line))
    else:
        for i in diag_file:
            with open(i, 'r', encoding='utf-8') as f1:
                for line in f1.readlines():
                    data.append(json.loads(line))

    with open(data_path + 'id2symptom.json', 'r', encoding='utf-8') as f2:
        id2symptom = json.loads(f2.read())
    with open(data_path + 'id2disease.json', 'r', encoding='utf-8') as f3:
        id2disease = json.loads(f3.read())
    

    symptom2id = {j:i for i,j in id2symptom.items()}
    disease2id = {j:i for i,j in id2disease.items()}

    return data, id2symptom, id2disease, symptom2id, disease2id

def sum_prob(prob_list):
    final_prob = prob_list[0]
    if len(prob_list) > 1:
        for i in range(1, len(prob_list)):
            if final_prob > 0 and prob_list[i] > 0:
                final_prob = final_prob + prob_list[i] - final_prob * prob_list[i]
            elif final_prob < 0 and prob_list[i] < 0:
                final_prob = final_prob + prob_list[i] + final_prob * prob_list[i]
            else:
                final_prob = (final_prob + prob_list[i]) / (1 - min(abs(final_prob), abs(prob_list[i])))            
    return final_prob