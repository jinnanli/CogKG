import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import numpy as np
from utils import *
from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn

DEVICE = 'cuda'

class NeuroNet(torch.nn.Module):
        def __init__(self, symptom_num, embedding_dim, disease_num, rule_num, rel_embeddings, dise_embeddings,pretrained_weights, weights_sklearn=None, bias_sklearn=None):
            super(NeuroNet, self).__init__()
            self.rule_num = rule_num # 
            self.symptom_num = symptom_num # 93
            self.embedding_dim  = embedding_dim # 512
            self.disease_num = disease_num # 12
            self.embeddings = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
            self.rel_embed_layer = nn.Embedding.from_pretrained(rel_embeddings, freeze=False) 
            self.dise_embed_layer = nn.Embedding.from_pretrained(dise_embeddings, freeze=False) 
            self.rel_embeddings = self.rel_embed_layer.weight
            self.dise_embeddings = self.dise_embed_layer.weight
            # self.rel_embeddings = rel_embeddings # (3, 512)
            # self.dise_embeddings = dise_embeddings # (12, 512)
            self.fc = nn.Linear(self.disease_num * (len(rel_embeddings) * self.symptom_num + self.rule_num), self.disease_num)
            if weights_sklearn is not None:
                self.fc.weight.data = weights_sklearn
            if bias_sklearn is not None:
                self.fc.bias.data = bias_sklearn            

            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs, rule_features):

            embeds = self.embeddings.weight # (93, 512)

            embeds_add_rel = torch.unsqueeze(embeds, 1) # (93, 3, 512)
            embeds_add_rel = embeds_add_rel.repeat((1, len(self.rel_embeddings), 1)) # (93, 3, 512)

            mask = torch.zeros_like(embeds_add_rel) # (93, 3, 512)

            for idx in torch.nonzero(inputs).squeeze(1): # activate with input symptoms
                mask[idx] = torch.ones_like(embeds_add_rel[0])

            embeds_add_rel += self.rel_embeddings
            embeds_add_rel *= mask

            # for idx, rel_emb in enumerate(self.rel_embeddings):
            #     embeds_add_rel[idx] += rel_emb
            #     embeds_add_rel[idx] *= mask

            embeds_add_rel = embeds_add_rel.reshape((-1, embeds_add_rel.shape[-1])) # [3 * 93, 512]

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            kg_features = torch.zeros((self.dise_embeddings.shape[0], embeds_add_rel.shape[0])).to(DEVICE)
            for idx, dise_embed in enumerate(self.dise_embeddings):
                # print(dise_embed.size())
                dise_embed = torch.unsqueeze(dise_embed, 0)
                # print(dise_embed.size())
                # print(embeds_add_rel.size())
                tmp = cos(embeds_add_rel, dise_embed)
                # print(tmp.size())
                kg_features[idx] = tmp
            
            # print(kg_features[0])
            # print(inputs)

            # print(kg_features.shape) # (12, 279)
            # print(rule_features.shape) # (12, 182)

            hidden = torch.cat((rule_features, kg_features), dim=1) # (12, 461)

            output = self.fc(hidden.view((1, -1))) 
            output = self.sigmoid(output) # (12, )
            return output


def check_triple(in_no, V_knw, id2embed_ent, id2embed_rel, id2rel):
    V_knw_tmp = V_knw.copy()
    in_no_prob = []
    for knw in V_knw_tmp:
        if V_knw_tmp[knw] > 0:
            for idx_rel, vec_rel in enumerate(id2embed_rel):
                if id2rel[idx_rel] in ['focus_of', 'associated_with', 'temporally_related_to']:
                    vec_sub = id2embed_ent[knw]
                    # vec_rel = id2embed_rel[rel]
                    vec1 = np.add(vec_sub, vec_rel)
                    vec2 = id2embed_ent[in_no]

                    CosSim = float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) * V_knw_tmp[knw]
                    # print(f'dist:{dist}')
                    in_no_prob.append(CosSim)

    return max(in_no_prob) if len(in_no_prob) > 0 else 0



def test_model(model, data):

    # sklaern_cls_id_map = {0:'C0004096', 1:'C0009763', 2:'C0011615', 3:'C0014335', 4:'C0014868', 5:'C0024894', 6:'C0029878', 7:'C0032285', 8:'C0035455', 9:'C0040147', 10:'C0876926', 11:'C1279369'}
    # _, _, _, _, dise2id = load_data('/home/weizhepei/workspace/CogKG_Neuro/data/diagnose/aligned/', 'train')

    model.eval()

    y_test = []
    y_pred = []
    cls_prob = []
    for i in data:
        inputs, targets, rule_features= i
        inputs = torch.tensor(inputs).to(DEVICE)
        rule_features = torch.FloatTensor(rule_features).to(DEVICE)
        y_test.extend(targets) 
        output = model(inputs, rule_features)
        cls_prob.append(list((output.data.cpu().detach().numpy()[0])))
        output = list(torch.max(output.data.cpu(), 1).indices.detach().numpy())
        y_pred.extend(output)
        # y_pred.extend([int(dise2id[sklaern_cls_id_map[i]]) for i in output])

    mrr, hits_1, hits_2 = get_rank_score(cls_prob, y_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_p = precision_score(y_test, y_pred, average='macro')
    macro_r = recall_score(y_test, y_pred, average='macro')

    print(f'macro p: {macro_p}, r: {macro_r}, f1: {macro_f1}')

    hits_1_score = hits_1
    hits_2_score = hits_2
    mrr = mrr

    print(f'Hits@1: {hits_1_score}')  
    print(f'Hits@2: {hits_2_score}')  
    print(f'MRR: {mrr}\n')

    PERFORMANCE = {'macro_p':macro_p, 'macro_r':macro_r, 'macro_f1':macro_f1, 'Hits@1':hits_1_score, 'Hits@2':hits_2_score, 'MRR':mrr}

    with open(f'../PERFORMANCE_CogRepre_NeuroNet.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(PERFORMANCE, ensure_ascii=False, indent=4))
    
    return hits_1_score

def get_rule_feature(id2dise, dise2id, ent2id, id2rel, id2ent, id2embed_ent, id2embed_rel, rules, symptoms):
    rule_feature_map = []
    for idx, r in enumerate(rules):
        symptoms_ids = {ent2id[i]:{'True':1.0, 'False':-1.0}[j] for i,j in symptoms.items()}
        rule_feature = np.zeros(len(id2dise))
        premise = [ent2id[i] for i in rules[r][0]]
        conclusion = rules[r][1][0]
        confidence = rules[r][2]
        rule_feature[int(dise2id[conclusion])] = confidence
    
        coefficient = 1 # early reject = -1, rule fired = 1, otherwise, calculate via link prediction

        early_break = False
        for i in premise:
            if i in symptoms_ids and symptoms_ids[i] == -1:
                early_break = True
                break

        if early_break:
            coefficient = -1
        elif len(set(premise) - set(symptoms_ids.keys())) == 0: # all premises are satisfied
            coefficient = 1
        else: # do link prediction for premise
            for in_no in set(premise) - set(symptoms_ids.keys()):
                prob = check_triple(in_no, symptoms_ids, id2embed_ent, id2embed_rel, id2rel)
                symptoms_ids[in_no] = prob
            coefficient = min([symptoms_ids[i] for i in premise])
        
        rule_feature_map.append(rule_feature * coefficient)
    return np.transpose(np.asarray(rule_feature_map))


def prepare_data(data_path, rule_dict, ent2id, id2ent, rel2id, id2rel, embeddings, split):
    
    id2embed_ent, id2embed_rel = embeddings.solver.values()

    spec_relations = ['focus_of', 'associated_with', 'temporally_related_to']
    data, id2symp, id2dise, symp2id, dise2id = load_data(data_path, split)
    
    id2disease_sklearn_lr = {0:'C0004096', 1:'C0009763', 2:'C0011615', 3:'C0014335', 4:'C0014868', 5:'C0024894', 6:'C0029878', 7:'C0032285', 8:'C0035455', 9:'C0040147', 10:'C0876926', 11:'C1279369'}
    disease2id_sklearn_lr = {j:i for i,j in id2disease_sklearn_lr.items()}

    processed_data = []
    for sample in tqdm(data):
        input = [0] * len(id2symp) 
        for k,v in sample['symptoms'].items():
            if v == 'True':
                input[int(symp2id[k])] = 1

        # if np.sum(input) == 0:
        #     print(sample)
        
        # output = [int(dise2id[sample['disease']])]
        output = [int(disease2id_sklearn_lr[sample['disease']])]

        rule_feature = get_rule_feature(id2dise, dise2id, ent2id, id2rel, id2ent, id2embed_ent, id2embed_rel, rule_dict, sample['symptoms'])

        processed_data.append((input, output, rule_feature))

    if split == 'train':
        symptom_embeddings = []
        for k,v in id2symp.items(): # id -> symptom
            symptom_embeddings.append(id2embed_ent[ent2id[v]])
        symptom_embeddings = np.asarray(symptom_embeddings)

        dise_embeddings = []
        for k,v in id2dise.items():
            dise_embeddings.append(id2embed_ent[ent2id[v]])
        dise_embeddings = np.asarray(dise_embeddings)

        rel_embeddings = []
        for rel in spec_relations:
            rel_embeddings.append(id2embed_rel[rel2id[rel]])
        rel_embeddings = np.asarray(rel_embeddings)

        print(f'symptom_embeddings: {len(symptom_embeddings)}, {len(symptom_embeddings[0])}')
        print(f'dise_embeddings: {len(dise_embeddings)}, {len(dise_embeddings[0])}')
        print(f'rel_embeddings: {len(rel_embeddings)}, {len(rel_embeddings[0])}')

        return np.asarray(processed_data, dtype='object'), symptom_embeddings, dise_embeddings, rel_embeddings
    else:
        return np.asarray(processed_data, dtype='object')

def main():
    import time
    start_time = time.time()

    CogKG_path = './CogKG_Neuro_eng/'
    data_path = CogKG_path + 'data/diagnose/aligned/'
    rule_path = CogKG_path + 'data/rule/disease_rule/'
    KG_path = CogKG_path + "data/KG/"

    saved_train_data = CogKG_path + 'data_train.npy'
    saved_valid_data = CogKG_path + 'data_valid.npy'
    saved_test_data = CogKG_path + 'data_test.npy'

    rule_dict = load_rules(rule_path)
    ent2id, id2ent, rel2id, id2rel, embeddings = load_KG(KG_path)

    if os.path.exists(saved_train_data):
        data_train, symptom_embeddings, dise_embeddings, rel_embeddings = np.load(saved_train_data, allow_pickle=True)
    else:
        data_train, symptom_embeddings, dise_embeddings, rel_embeddings = prepare_data(data_path, rule_dict, ent2id, id2ent, rel2id, id2rel, embeddings, split='train')
        np.save(saved_train_data, (data_train, symptom_embeddings, dise_embeddings, rel_embeddings))

    if os.path.exists(saved_valid_data):
        data_valid = np.load(saved_valid_data, allow_pickle=True)
    else:
        data_valid = prepare_data(data_path, rule_dict, ent2id, id2ent, rel2id, id2rel, embeddings, split='valid')
        np.save(saved_valid_data, data_valid)

    if os.path.exists(saved_test_data):
        data_test = np.load(saved_test_data, allow_pickle=True)
    else:
        data_test = prepare_data(data_path, rule_dict, ent2id, id2ent, rel2id, id2rel, embeddings, split='test')
        np.save(saved_test_data, data_test)

    print(f'Traning Size:{len(data_train)}')
    print(f'Valid Size:{len(data_valid)}')
    print(f'Test Size:{len(data_test)}')

    weights_sklearn = np.load('weights.npy')
    bias_sklearn = np.load('bias.npy')
    model = NeuroNet(len(symptom_embeddings), len(symptom_embeddings[0]), len(dise_embeddings), len(rule_dict), torch.FloatTensor(rel_embeddings).to(DEVICE), torch.FloatTensor(dise_embeddings).to(DEVICE), pretrained_weights=torch.FloatTensor(symptom_embeddings).to(DEVICE), weights_sklearn=torch.FloatTensor(weights_sklearn).to(DEVICE), bias_sklearn=torch.FloatTensor(bias_sklearn).to(DEVICE))

    # model = NeuroNet(len(symptom_embeddings), len(symptom_embeddings[0]), len(dise_embeddings), len(rule_dict), torch.FloatTensor(rel_embeddings).to(DEVICE), torch.FloatTensor(dise_embeddings).to(DEVICE), pretrained_weights=torch.FloatTensor(symptom_embeddings).to(DEVICE))

    model = model.to(DEVICE)

    for para in model.fc.parameters():
        para.requires_grad = False

    # Trainable Layers
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6) #5e-6 

    loss_function = nn.CrossEntropyLoss()

    BEST_SCORE = 0
    BEST_EPOCH = 0
    ACCUM_STEPS = 1 # [1, 16, 32, 64, 128, 512]:
    EPOCH_CNT = 1
    TOLERENCE_CNT = 0
    # Run the training loop
    # FEATURES = []
    while True:
        print(f'Starting epoch {EPOCH_CNT}')
        # Set current loss value
        current_loss = 0.0
        
        shuffle_idx = np.random.permutation(np.arange(len(data_train)))

        # Iterate over the DataLoader for training data
        for i, data in enumerate(data_train[shuffle_idx], 0):
        # for i, data in enumerate(data_train, 0):
            # print(f'sample-{i}')
            # Get inputs
            inputs, targets, rule_features= data
            inputs = torch.tensor(inputs).to(DEVICE)
            targets = torch.tensor(targets).to(DEVICE)
            rule_features = torch.FloatTensor(rule_features).to(DEVICE)
            # Perform forward pass
            outputs = model(inputs, rule_features)
            
            # FEATURES.append(hidden.cpu().numpy().astype(np.float64))
            # Compute loss
            loss = loss_function(outputs, targets)

            loss = loss / ACCUM_STEPS
            # Perform backward pass
            loss.backward()            
            
            if (i + 1) % ACCUM_STEPS == 0:
                # Perform optimization
                optimizer.step()
                # Zero the gradients
                optimizer.zero_grad()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after samples %5d: %.5f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0
        
        # np.save('./features.npy', FEATURES)

        cur_score = test_model(model, data_valid)

        if cur_score > BEST_SCORE:
            print(f'New Best Score Found in epoch {EPOCH_CNT}!\n')
            TOLERENCE_CNT = 0
            BEST_SCORE = cur_score
            BEST_EPOCH = EPOCH_CNT
            torch.save(model, f'saved_best_model_{ACCUM_STEPS}.pt')
        else:
            TOLERENCE_CNT += 1
        
        # torch.save(model, f'saved_best_model_embed_{ACCUM_STEPS}.pt')

        # Early Stop
        if TOLERENCE_CNT > 5:
            break
        
        # if EPOCH_CNT == 200:
        #     break
        
        EPOCH_CNT += 1

    # Process is complete.
    print(f'Training process ({BEST_EPOCH} epoch, accum steps {ACCUM_STEPS}) has finished. Best valid hits@1:{BEST_SCORE}')

    end_time = time.time()
    print(f'time cost:{(end_time - start_time) / 3600.0} hrs')

    well_trained_model = torch.load(f'saved_best_model_{ACCUM_STEPS}.pt')
    # well_trained_model = torch.load(f'best_model.pt') # 5e-5, 2 epoch
    # well_trained_model = torch.load(f'saved_best_model_embed_{ACCUM_STEPS}.pt')
    # well_trained_model = model

    print(f'Train Set Performance:')
    test_model(well_trained_model, data_train)
    print(f'Valid Set Performance:')
    test_model(well_trained_model, data_valid)
    print(f'Test Set Performance:')
    test_model(well_trained_model, data_test)

if __name__ == '__main__':
    main()