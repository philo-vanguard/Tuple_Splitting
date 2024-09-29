import copy

import faiss
import pandas as pd
import numpy as np
from operator import itemgetter
import torch


# ------------------------------------- below are for impute one tuple, i.e., Series by Md -------------------------------------
def CandidateAwardAll(person_ID, attribute, DisAM_dict, DisAM_id_dict, person_info_all):
    candidate_award = []
    candidate_person_list = DisAM_dict['person'][DisAM_id_dict[person_ID]].split(',')
    for i in candidate_person_list:
        id = int(i)
        if (person_info_all.__contains__(id)):
            if (person_info_all[id].__contains__(attribute)):
                candidate_award.extend([int(j) for j in person_info_all[id][attribute].split(',') if j != ''])
    candidate_award_all = list(set(candidate_award))
    person_award_all = [int(j) for j in person_info_all[person_ID][attribute].split(',') if j != '']
    labels = np.zeros(len(candidate_award_all))
    for i in person_award_all:
        labels[candidate_award_all.index(i)] = 1
    # return candidate_award_all,person_award_all
    return candidate_award_all, labels


def wiki_embed_load(List, wikidata_dic, wikidata_embed):
    if len(List) > 1:
        List = [int(i) for i in List]
        retrieve = itemgetter(*List)(wikidata_dic)
        return wikidata_embed[list(retrieve)]
    elif len(List) == 1:
        return wikidata_embed[wikidata_dic[int(List[0])]].reshape((-1, 200))
    else:
        return np.zeros((200)).astype('float32').reshape((-1, 200))


def impute_Md(tuple_in_series, DisAM_dict, DisAM_id_dict, item_dict, person_info_all, wikidata_embed, wikidata_dic, item_dict_reverse):
    tuple_in_series = tuple_in_series.drop("full_name")
    col = pd.DataFrame(tuple_in_series).T.columns
    length = len(col) - 1
    id = int(tuple_in_series['id'])
    A_ = tuple_in_series[~tuple_in_series.isnull()]

    A_empty = A_[A_ == '-']
    A_fill = A_[A_ != '-']

    mrr_1 = []
    mrr_result = []
    for i in A_fill.iteritems():
        if i[0] in col[:10]:
            A_fill[i[0]] = item_dict_reverse[i[0].replace("_", " ")][i[1]]

    for i in A_empty.iteritems():
        if i[0] in col[:length]:
            candidate, y_true = CandidateAwardAll(id, i[0].replace("_", " "), DisAM_dict, DisAM_id_dict, person_info_all)
            # print(candidate)
            candidate_embed = wiki_embed_load(candidate, wikidata_dic, wikidata_embed)
            entity_embed = np.mean(wiki_embed_load(A_fill.values, wikidata_dic, wikidata_embed), axis=0).reshape(-1, 200)
            # entity_embed_2 = wikidata_embed[wikidata_dic[id]].reshape(-1,200)
            # entity_embed = np.mean(np.vstack((entity_embed,entity_embed_2)),axis=0).reshape(-1,200)
            index = faiss.IndexFlatIP(200)
            index.add(candidate_embed)
            DPScores_OP, neighbors_OP = index.search(entity_embed.reshape((-1, 200)), candidate_embed.shape[0])
            # print(DPScores_OP,neighbors_OP ,y_true)
            imputation = neighbors_OP[0][0]
            if len(y_true) > 0:
                if y_true[imputation] == 1:
                    mrr_1.append(1)
                else:
                    mrr_1.append(0)
                mrr_result.append(
                    [i[0], item_dict[candidate[imputation]], DPScores_OP[0][0] / sum(DPScores_OP[0])])
            else:
                mrr_1.append(0)
            # mrr_result.append([i[0],self.item_dict[candidate[imputation]],DPScores_OP[0][0]/sum(DPScores_OP[0])])
            mrr_result.append([i[0], '', 0])

    return mrr_1, mrr_result  # first: whether correctly impute the missing attribute values; second: attribute name, imputed value, and confidence


# ------------------------------------- below are for impute multi tuples, i.e., DataFrame by Md -------------------------------------
def TransAttribute_vector(row, wikidata_embed, wikidata_dic, item_dict_reverse):
    temp = []
    for x, y in row.iteritems():
        if (not pd.isnull(y)) and (y != '-'):
            temp.append(wikidata_dic[item_dict_reverse[x][y]])
        else:
            temp.append(411211)
    # vector = wikidata_embed[temp].reshape((10,200))
    vector = wikidata_embed[temp].reshape((2000))
    return vector


def Confidence(array):
    a = np.array(array)
    vary = np.max(a)-np.min(a)
    a = a - np.min(a)
    score = np.max(a)/(np.sum(a)+0.00000001)
    return score


def CandidateAwardAll_batch(person_ID, attribute, DisAM_dict, DisAM_id_dict, person_info_all):
    candidate_award = []
    candidate_person_list = DisAM_dict['person'][DisAM_id_dict[person_ID]].split(',')
    for i in candidate_person_list:
        id = int(i)
        if (person_info_all.__contains__(id)):
            if (person_info_all[id].__contains__(attribute)):
                candidate_award.extend([int(j) for j in person_info_all[id][attribute].split(',') if j != ''])
    candidate_award_all = list(set(candidate_award))
    person_award_all = [int(j) for j in person_info_all[person_ID][attribute].split(',') if j != '']
    labels = np.zeros(len(candidate_award_all))
    for i in person_award_all:
        labels[candidate_award_all.index(i)] = 1
    # return candidate_award_all,person_award_all
    return candidate_award_all, labels


def wiki_embed_load_batch(List, wikidata_embed, wikidata_dic):
    if len(List) > 1:
        List = [int(i) for i in List]
        retrieve = itemgetter(*List)(wikidata_dic)
        return wikidata_embed[list(retrieve)]
    elif len(List) == 1:
        return wikidata_embed[wikidata_dic[int(List[0])]].reshape((-1, 200))
    else:
        return np.zeros((200)).astype('float32').reshape((-1, 200))


def impute_Md_batch(df, missing_value, DisAM_dict, DisAM_id_dict, item_dict, person_info_all, wikidata_embed, wikidata_dic, item_dict_reverse, avg_pool, model_load):
    wiki_split = copy.deepcopy(df)
    wiki_split = wiki_split.where(wiki_split.notnull(), '-')
    old_columns = wiki_split.columns.tolist()
    for index, value in enumerate(old_columns):
        old_columns[index] = value.replace("_", " ")
    wiki_split.columns = old_columns
    wiki_split['Index'] = wiki_split.index

    wiki_split_embed = df.iloc[:, :10].parallel_apply(lambda row: TransAttribute_vector(row, wikidata_embed, wikidata_dic, item_dict_reverse), axis=1, result_type='expand')
    wiki_split_embed = np.array(wiki_split_embed).reshape((-1, 10, 200))
    print('Graph Embedding Load Complete')
    wiki_split_embed_result = np.zeros((len(wiki_split_embed), 1, 200))
    print('Inference Transformers')

    for i in range(np.int64(np.ceil(len(wiki_split_embed) / 5000))):  ## Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
        vector = torch.from_numpy(wiki_split_embed[i * 5000: min((i + 1) * 5000, len(wiki_split_embed))]).to(device)  ## Device: cuda0, do not change
        vector = vector.to(torch.float32)  ## capable of inference with torch.float32
        result = model_load.forward(vector).detach()  ## inference model, detach from grad()
        for j in range(i * 5000, min((i + 1) * 5000, len(wiki_split_embed)), 1):
            wiki_split_embed_result[j] = avg_pool(result[j - i * 5000].reshape((1, 10, 200)))[0].cpu().numpy()
        torch.cuda.empty_cache()

    print('Inference Complete, Starting parallel imputation')

    # pandarallel.initialize()
    def impute_Md(tuple_in_series):
        Index = int(tuple_in_series['Index'])
        col = pd.DataFrame(tuple_in_series).T.columns
        length = len(col) - 1
        id = int(tuple_in_series['id'])
        A_full = tuple_in_series[:10]
        tuple_in_series = tuple_in_series[:10]

        A_ = tuple_in_series[~tuple_in_series.isnull()]

        A_empty = A_[A_ == '-']
        A_fill = A_[A_ != '-']

        mrr_1 = []
        mrr_result = []
        for i in A_fill.iteritems():
            if i[0] in col[:10]:
                A_fill[i[0]] = item_dict_reverse[i[0]][i[1]]

        for i in A_empty.iteritems():
            if i[0] != missing_value.replace("_", " "):
                continue
            if i[0] in col[:10]:
                # print(i[0])
                candidate, y_true = CandidateAwardAll_batch(id, i[0], DisAM_dict, DisAM_id_dict, person_info_all)
                candidate_embed = wiki_embed_load_batch(candidate, wikidata_embed, wikidata_dic)

                candidate_person_list = [int(j) for j in DisAM_dict['person'][DisAM_id_dict[id]].split(',')]
                candidate_person_embed = wiki_embed_load_batch(candidate_person_list, wikidata_embed, wikidata_dic)

                entity_embed = torch.from_numpy(
                    wiki_split_embed_result[Index].reshape(1, 200))  ## 取代TransAttribute，拿到Transformer的结果
                candidate_person_embed_torch = torch.from_numpy(candidate_person_embed)
                Score = torch.argmax(torch.cosine_similarity(entity_embed, candidate_person_embed_torch)).numpy()
                entity_embed = candidate_person_embed[Score]

                index = faiss.IndexFlatIP(200)
                index.add(candidate_embed)
                DPScores_OP, neighbors_OP = index.search(entity_embed.reshape((-1, 200)), candidate_embed.shape[0])
                # print(DPScores_OP,neighbors_OP ,y_true)
                imputation = neighbors_OP[0][0]
                if len(y_true) > 0:  ## Have Candidate List
                    if (np.sum(y_true == 1) > 0):  ## Have True Value
                        if y_true[imputation] == 1:
                            mrr_1.append(1)
                        else:
                            mrr_1.append(0)
                    mrr_result.append([i[0], item_dict[candidate[imputation]], Confidence(DPScores_OP[0])])
                else:
                    mrr_result.append([i[0], '', 0])

        return mrr_1, mrr_result  # first: whether correctly impute the missing attribute values; second: attribute name, imputed value, and confidence

    wiki_split_result_right_ = wiki_split.parallel_apply(impute_Md, axis=1, result_type='expand')
    # old_columns = wiki_split_result_right_.columns.tolist()
    # print(old_columns)
    # for index, value in enumerate(old_columns):
    #     old_columns[index] = value.replace(" ", "_")
    # wiki_split_result_right_.columns = old_columns
    # wiki_split_result_right_.to_csv(arg_dict["output_splitting_tuples_path"], index=False)

    A_result = []
    for i in wiki_split_result_right_.iloc[:, 0].values:
        A_result.extend(i)
    print(np.mean(A_result), len(A_result))
    return wiki_split_result_right_
