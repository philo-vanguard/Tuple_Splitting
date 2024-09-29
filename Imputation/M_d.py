import faiss
import pandas as pd
import numpy as np
from operator import itemgetter
from CorrelationModel.Attentions import ReversePredictor
import torch
import copy
import importlib
from pandarallel import pandarallel
import time
import random
pd.set_option('mode.chained_assignment', None)

all_data_dir = "/tmp/tuple_splitting/dict_model/"

# Md model
class ImputationModel:
    def __init__(self):
        self.data_name = None
        self.if_parallel = False
        self.training_ratio = None
        self.mask_ratio = None

        self.avg_pool = None
        self.model_load = None
        self.device = None
        self.embedding_dim = None
        self.sequence_dim = None

        self.m_d_imputation_persons = None
        self.m_d_imputation_college = None
        self.m_d_imputation_imdb = None

        # for persons
        self.DisAM_dict = None
        self.DisAM_id_dict = None
        self.person_info_all = None
        self.item_dict = None
        self.wikidata_embed = None
        self.wikidata_dic = None
        self.item_dict_reverse = None

        # for imdb
        self.imdb_dict_reverse = None
        self.imdb_person_dict_reverse = None
        self.imdb_embedding_dict = None
        self.imdb_embed = None
        self.imdb_info_all = None
        self.imdb_person_dict = None
        self.DisAM_movie_dict = None
        self.weight = None
        self.embedding = None

        # for dblp
        self.dblp_embedding_dict = None
        self.dblp_embedding = None
        self.dblp_DisAM_dict = None
        self.dblp_dict = None
        self.dblp_person_name_dict_reverse = None
        self.dblp_person_name_dict_reverse_generate = None
        self.dblp_type_dict = None
        self.dblp_t5_embedding_dict = None

        # for college
        self.college_info_all = None
        self.college_merge_dict = None
        self.college_embedding_dict = None
        self.college_embed = None

    def load_model_Md(self, data_name, cuda, if_parallel, training_ratio, mask_ratio):
        self.data_name = data_name
        self.training_ratio = training_ratio
        self.mask_ratio = mask_ratio

        if self.data_name == "persons":
            if self.if_parallel is False:
                self.wikidata_embed = np.load(all_data_dir + 'dict/wikimedia/wikidata_embed_shrink.npy', mmap_mode='r')
                self.wikidata_dic = np.load(all_data_dir + 'dict/wikimedia/wikidata_dict_shrink_person.npy', allow_pickle=True).item()
                self.item_dict = np.load(all_data_dir + 'dict/wikimedia/item_dict_shrink.npy', allow_pickle=True).item()
                self.item_dict_reverse = np.load(all_data_dir + 'dict/wikimedia/item_dict_reverse_all.npy', allow_pickle=True).item()
                self.DisAM_dict = np.load(all_data_dir + 'dict/wikimedia/DisAM_dict.npy', allow_pickle=True).item()
                self.DisAM_id_dict = np.load(all_data_dir + 'dict/wikimedia/DisAM_id_dict.npy', allow_pickle=True).item()
                self.person_info_all = np.load(all_data_dir + 'dict/wikimedia/person_info_all_full.npy', allow_pickle=True).item()
                self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 200))
                self.device = torch.device("cuda:"+str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                if training_ratio == 1.0:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/wiki/M_d/100.ckpt').to(self.device)
                elif training_ratio == 0.2:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/wiki/M_d/20.ckpt').to(self.device)
                elif training_ratio == 0.4:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/wiki/M_d/40.ckpt').to(self.device)
                elif training_ratio == 0.6:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/wiki/M_d/60.ckpt').to(self.device)
                elif training_ratio == 0.8:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/wiki/M_d/80.ckpt').to(self.device)
                self.model_load = self.model_load.to(self.device)

        elif self.data_name == "imdb":
            if self.if_parallel is False:
                self.imdb_dict_reverse = np.load(all_data_dir + 'dict/IMDB/imdb_dict_reverse_all.npy', allow_pickle=True).item()
                self.imdb_person_dict_reverse = np.load(all_data_dir + 'dict/IMDB/imdb_person_dict_reverse.npy', allow_pickle=True).item()
                self.imdb_embedding_dict = np.load(all_data_dir + 'dict/IMDB/imdb_embedding_shrink_dict.npy', allow_pickle=True).item()
                self.imdb_embed = np.load(all_data_dir + 'dict/IMDB/imdb_embed_shrink.npy', mmap_mode='r')
                self.imdb_info_all = np.load(all_data_dir + 'dict/IMDB/imdb_info_all.npy', allow_pickle=True).item()
                self.imdb_person_dict = np.load(all_data_dir + 'dict/IMDB/imdb_person_dict.npy', allow_pickle=True).item()
                self.DisAM_movie_dict = np.load(all_data_dir + 'dict/IMDB/DisAM_movie_dict.npy', allow_pickle=True).item()
                self.device = torch.device("cuda:"+str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                self.embedding_dim = 200
                if training_ratio == 1.0:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/imdb/M_d/100.ckpt').to(self.device)
                elif training_ratio == 0.2:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/imdb/M_d/20.ckpt').to(self.device)
                elif training_ratio == 0.4:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/imdb/M_d/40.ckpt').to(self.device)
                elif training_ratio == 0.6:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/imdb/M_d/60.ckpt').to(self.device)
                elif training_ratio == 0.8:
                    self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/imdb/M_d/80.ckpt').to(self.device)
                self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, self.embedding_dim))

        elif self.data_name == "dblp":
            # self.dblp_embedding_dict = np.load(all_data_dir + 'dict/DBLP/dblp_embedding_dict.npy', allow_pickle=True).item()
            self.dblp_embedding = np.load(all_data_dir + 'dict/DBLP/dblp_embedding.npy')
            self.dblp_DisAM_dict = np.load(all_data_dir + 'dict/DBLP/dblp_DisAM_dict.npy', allow_pickle=True).item()
            self.dblp_dict = np.load(all_data_dir + 'dict/DBLP/dblp_dict_t5.npy',allow_pickle=True).item()
            # self.dblp_person_name_dict_reverse = np.load(all_data_dir + 'dict/DBLP/dblp_person_name_dict_reverse.npy', allow_pickle=True).item()
            self.dblp_person_name_dict_reverse_generate = np.load(all_data_dir + 'dict/DBLP/dblp_person_name_dict_reverse_generate.npy', allow_pickle=True).item()
            self.dblp_t5_embedding_dict = np.load(all_data_dir + 'dict/DBLP/dblp_t5_embedding_dict_str.npy', allow_pickle=True).item()
            self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
            self.dblp_type_dict = {
                'type': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                'publish': 'https://dblp.org/rdf/schema#publishedIn',
                'year': 'https://dblp.org/rdf/schema#yearOfPublication',
                'page': 'https://dblp.org/rdf/schema#pagination',
                'author': 'https://dblp.org/rdf/schema#authoredBy',
                'vol': 'vol',
                'title': 'https://dblp.org/rdf/schema#title',
                'ref': 'http://www.w3.org/2000/01/rdf-schema#label',
            }
            self.embedding_dim = 200
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, self.embedding_dim))
            if training_ratio == 1.0:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/DBLP/M_d/100.ckpt').to(self.device)
            elif training_ratio == 0.2:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/DBLP/M_d/20.ckpt').to(self.device)
            elif training_ratio == 0.4:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/DBLP/M_d/40.ckpt').to(self.device)
            elif training_ratio == 0.6:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/DBLP/M_d/60.ckpt').to(self.device)
            elif training_ratio == 0.8:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/DBLP/M_d/80.ckpt').to(self.device)

        elif self.data_name == "college":
            if self.if_parallel is False:
                self.college_info_all = np.load(all_data_dir + 'dict/college/college_info_all.npy', allow_pickle=True).item()
                self.college_merge_dict = np.load(all_data_dir + 'dict/college/college_merge_dict.npy', allow_pickle=True).item()
                self.college_embedding_dict = np.load(all_data_dir + 'dict/college/college_embedding_dict.npy', allow_pickle=True).item()
                self.college_embed = np.load(all_data_dir + 'dict/college/college_embed.npy')
                self.device = torch.device("cuda:"+str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                self.sequence_dim = 8
                self.embedding_dim = 200
                self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, self.embedding_dim))
            if training_ratio == 1.0:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/college/M_d/100.ckpt').to(self.device)
            elif training_ratio == 0.2:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/college/M_d/20.ckpt').to(self.device)
            elif training_ratio == 0.4:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/college/M_d/40.ckpt').to(self.device)
            elif training_ratio == 0.6:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/college/M_d/60.ckpt').to(self.device)
            elif training_ratio == 0.8:
                self.model_load = ReversePredictor.load_from_checkpoint(all_data_dir + 'models/college/M_d/80.ckpt').to(self.device)

    def impute_Md_batch(self, tuples_df, missing_attribute):
        if self.data_name == "persons":
            return self.impute_Md_batch_persons(tuples_df, missing_attribute)
        elif self.data_name == "imdb":
            return self.impute_Md_batch_imdb(tuples_df, missing_attribute, reverse_dict=True)
        elif self.data_name == "dblp":
            return self.impute_Md_batch_dblp(tuples_df, missing_attribute, reverse_dict=True)
        elif self.data_name == "college":
            return self.impute_Md_batch_college(tuples_df, missing_attribute)

    def impute_Md_batch_persons(self, tuples_df, missing_attribute):
        def TransAttribute_vector(row):
            temp = []
            for x, y in row.iteritems():
                if (not pd.isnull(y)) and (y != '-'):
                    temp.append(self.wikidata_dic[self.item_dict_reverse[x][y]])
                else:
                    temp.append(411211)
            vector = self.wikidata_embed[temp].reshape((2000))
            return vector

        def Confidence(array):
            a = np.array(array)
            vary = np.max(a) - np.min(a)
            a = a - np.min(a)
            score = np.max(a) / (np.sum(a) + 0.00000001)
            return score

        def CandidateAwardAll_batch(person_ID, attribute):
            candidate_award = []
            candidate_person_list = self.DisAM_dict['person'][self.DisAM_id_dict[person_ID]].split(',')
            for i in candidate_person_list:
                id = int(i)
                if self.person_info_all.__contains__(id):
                    if self.person_info_all[id].__contains__(attribute):
                        candidate_award.extend([int(j) for j in self.person_info_all[id][attribute].split(',') if j != ''])
            candidate_award_all = list(set(candidate_award))
            person_award_all = [int(j) for j in self.person_info_all[person_ID][attribute].split(',') if j != '']
            labels = np.zeros(len(candidate_award_all))
            for i in person_award_all:
                labels[candidate_award_all.index(i)] = 1
            return candidate_award_all, labels

        def wiki_embed_load_batch(List):
            if len(List) > 1:
                List = [int(i) for i in List]
                retrieve = itemgetter(*List)(self.wikidata_dic)
                return self.wikidata_embed[list(retrieve)]
            elif len(List) == 1:
                return self.wikidata_embed[self.wikidata_dic[int(List[0])]].reshape((-1, 200))
            else:
                return np.zeros((200)).astype('float32').reshape((-1, 200))

        wiki_split = copy.deepcopy(tuples_df)
        wiki_split = wiki_split.where(wiki_split.notnull(), '-')
        old_columns = wiki_split.columns.tolist()
        for index, value in enumerate(old_columns):
            old_columns[index] = value.replace("_", " ")
        wiki_split.columns = old_columns
        wiki_split = wiki_split.reset_index(drop=True)
        wiki_split['Index'] = wiki_split.index

        results = None
        if self.if_parallel is False:
            wiki_split_embed = wiki_split.iloc[:, :10].apply(TransAttribute_vector, axis=1, result_type='expand')
            wiki_split_embed = np.array(wiki_split_embed).reshape((-1, 10, 200))
            wiki_split_embed_result = np.zeros((len(wiki_split_embed), 1, 200))

            for i in range(np.int64(np.ceil(len(wiki_split_embed) / 5000))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(wiki_split_embed[i * 5000: min((i + 1) * 5000, len(wiki_split_embed))]).to(self.device)
                vector = vector.to(torch.float32)  # capable of inference with torch.float32
                result = self.model_load.forward(vector).detach()  # inference model, detach from grad()
                for j in range(i * 5000, min((i + 1) * 5000, len(wiki_split_embed)), 1):
                    wiki_split_embed_result[j] = self.avg_pool(result[j - i * 5000].reshape((1, 10, 200)))[0].cpu().numpy()
                torch.cuda.empty_cache()

            def impute_Md(tuple_in_series):
                Index = int(tuple_in_series['Index'])
                col = pd.DataFrame(tuple_in_series).T.columns
                id = int(tuple_in_series['id'])
                tuple_in_series = tuple_in_series[:10]

                A_ = tuple_in_series[~tuple_in_series.isnull()]

                A_empty = A_[A_ == '-']
                A_fill = A_[A_ != '-']

                mrr_1 = []
                mrr_result = []
                for i in A_fill.iteritems():
                    if i[0] in col[:10]:
                        A_fill[i[0]] = self.item_dict_reverse[i[0]][i[1]]

                for i in A_empty.iteritems():
                    if i[0] != missing_attribute.replace("_", " "):
                        continue
                    if i[0] in col[:10]:
                        candidate, y_true = CandidateAwardAll_batch(id, i[0])
                        candidate_embed = wiki_embed_load_batch(candidate)

                        candidate_person_list = [int(j) for j in self.DisAM_dict['person'][self.DisAM_id_dict[id]].split(',')]
                        candidate_person_embed = wiki_embed_load_batch(candidate_person_list)

                        entity_embed = torch.from_numpy(wiki_split_embed_result[Index].reshape(1, 200))  ## 取代TransAttribute，拿到Transformer的结果
                        candidate_person_embed_torch = torch.from_numpy(candidate_person_embed)
                        Score = torch.argmax(torch.cosine_similarity(entity_embed, candidate_person_embed_torch)).numpy()
                        entity_embed = candidate_person_embed[Score]

                        index = faiss.IndexFlatIP(200)
                        index.add(candidate_embed)
                        DPScores_OP, neighbors_OP = index.search(entity_embed.reshape((-1, 200)), candidate_embed.shape[0])
                        imputation = neighbors_OP[0][0]
                        if len(y_true) > 0:  ## Have Candidate List
                            if np.sum(y_true == 1) > 0:  ## Have True Value
                                if y_true[imputation] == 1:
                                    mrr_1.append(1)
                                else:
                                    mrr_1.append(0)
                            mrr_result.append([i[0], self.item_dict[candidate[imputation]], Confidence(DPScores_OP[0])])
                        else:
                            mrr_result.append([i[0], '', 0])

                return mrr_1, mrr_result  # first: whether correctly impute the missing attribute values; second: attribute name, imputed value, and confidence

            wiki_split_result_right_ = wiki_split.apply(impute_Md, axis=1, result_type='expand')
            results = wiki_split_result_right_[1].values
            results = [i[0][1] if len(i) > 0 else "" for i in results]

        return results  # a list of values of str type

    def impute_Md_batch_imdb(self, tuples_df, missing_attribute, reverse_dict=True):
        def LimitMap(row):
            col_limit = pd.DataFrame(row).T.columns
            title = row['title']
            for x, y in row.iteritems():
                if (not pd.isnull(y)) and (y != '-'):
                    if x in ['actor', 'actress', 'writer', 'producer', 'director']:
                        if self.imdb_dict_reverse.__contains__(title):
                            if self.imdb_dict_reverse[title].__contains__(y):
                                row[x] = self.imdb_dict_reverse[title][y]
                            else:
                                try:
                                    row[x] = self.imdb_person_dict_reverse[y]
                                except:
                                    row[x] = y
                        else:
                            try:
                                row[x] = self.imdb_person_dict_reverse[y]
                            except:
                                row[x] = y
            return row

        def Confidence(array):
            a = np.array(array)
            vary = np.max(a) - np.min(a)
            a = a - np.min(a)
            score = np.max(a) / (np.sum(a) + 0.00000001)
            return score

        def TransIndex_imdb_vector(row):
            temp = []
            for x,y in row.iteritems():
                if (not pd.isnull(y)) and (y!='-'):
                    if(isinstance(y,float)):
                        temp.append(self.imdb_embedding_dict[str(int(float(y)))])
                    elif(isinstance(y,str)):
                        try:
                            temp.append(self.imdb_embedding_dict[y.strip().replace(' ','_')])
                        except:
                            temp.append(2447210)
                else:
                    temp.append(2447210)
            vector = self.imdb_embed[temp].reshape((2000))
            return vector

        def CandidateAwardAllImdb(person, attribute):
            candidate_award = []
            candidate_person_list = self.DisAM_movie_dict[person]
            for i in candidate_person_list:
                id = i
                if attribute == 'title':
                    candidate_award.extend(self.imdb_info_all[id][attribute].split('||'))
                elif self.imdb_info_all[id][attribute] != '':
                    candidate_award.extend(self.imdb_info_all[id][attribute].split(','))
            candidate_award_all = list(set(candidate_award))
            if attribute == 'title':
                person_award_all = self.imdb_info_all[person][attribute].split('||')
            else:
                person_award_all = self.imdb_info_all[person][attribute].split(',')

            labels = np.zeros(len(candidate_award_all))
            for i in person_award_all:
                try:
                    labels[candidate_award_all.index(i)] = 1
                except:
                    labels = labels
            candidate_award_all = [j.strip().replace(' ', '_') for j in candidate_award_all]
            return candidate_award_all, labels

        def imdb_embed_load(List):
            List = [i.strip().replace(' ', '_') for i in List]
            if len(List) > 1:
                retrieve = itemgetter(*List)(self.imdb_embedding_dict)
                return self.imdb_embed[list(retrieve)]
            elif len(List)==1:
                return self.imdb_embed[self.imdb_embedding_dict[List[0]]].reshape((-1, 200))
            else:
                return np.zeros((1,200)).astype('float32')

        def m_d_imputation(df_tuples):
            df = copy.deepcopy(df_tuples)
            if reverse_dict:
                if self.if_parallel is True:
                    df = df.parallel_apply(LimitMap, axis=1)
                else:
                    df = df.apply(LimitMap, axis=1)
            df = df.fillna('-')
            df = df.reset_index(drop=True)
            df['Index'] = df.index
            imdb_split = df

            imdb_split_embed = df.iloc[:, :10].apply(TransIndex_imdb_vector, axis=1, result_type='expand')
            imdb_split_embed = np.array(imdb_split_embed).reshape((-1, 10, self.embedding_dim))
            imdb_split_embed_result = np.zeros((len(imdb_split_embed), 1, self.embedding_dim))

            for i in range(np.int64(np.ceil(len(imdb_split_embed) / 5000))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(imdb_split_embed[i * 5000: min((i + 1) * 5000, len(imdb_split_embed))]).to(self.device)
                vector = vector.to(torch.float32)  # capable of inference with torch.float32
                result = self.model_load.forward(vector).detach()  # inference model, detach from grad()
                for j in range(i * 5000, min((i + 1) * 5000, len(imdb_split_embed)), 1):
                    imdb_split_embed_result[j] = self.avg_pool(result[j - i * 5000].reshape((1, 10, self.embedding_dim)))[0].cpu().numpy()
                torch.cuda.empty_cache()

            def mrr_test_movie(A_):
                col = pd.DataFrame(A_).T.columns
                # A_ = A.iloc[-3,:]
                id = A_['id']
                Index = A_['Index']

                A_ = A_[:-2]
                A_full = A_
                A_ = A_[~A_.isnull()]

                A_empty = A_[A_ == '-']
                A_fill = A_[A_ != '-']

                mrr_1 = []
                mrr_2 = []
                mrr_result = []

                # skip Title temporarily
                for i in A_empty.iteritems():
                    if i[0] != missing_attribute:
                        continue

                    if i[0] in col[:10]:
                        candidate, y_true = CandidateAwardAllImdb(id, i[0])
                        candidate_embed = imdb_embed_load(candidate)

                        candidate_person_list = self.DisAM_movie_dict[id]
                        candidate_person_embed = imdb_embed_load(candidate_person_list)

                        entity_embed = torch.from_numpy(imdb_split_embed_result[Index].reshape(1, self.embedding_dim))  ## 取代TransAttribute，拿到Transformer的结果
                        candidate_person_embed_torch = torch.from_numpy(candidate_person_embed)
                        Score = torch.argmax(torch.cosine_similarity(entity_embed, candidate_person_embed_torch)).numpy()
                        entity_embed = candidate_person_embed[Score]

                        if id == candidate_person_list[Score]:
                            mrr_2.append(1)
                        else:
                            mrr_2.append(0)

                        if len(candidate) > 0:
                            index = faiss.IndexFlatIP(self.embedding_dim)
                            index.add(candidate_embed)
                            DPScores_OP, neighbors_OP = index.search(entity_embed.reshape((-1, self.embedding_dim)), len(candidate))
                            imputation = neighbors_OP[0][0]
                            if len(y_true) > 0:  # Have Candidate List
                                if np.sum(y_true == 1) > 0:  # Have True Value
                                    if y_true[imputation] == 1:
                                        mrr_1.append(1)
                                    else:
                                        mrr_1.append(0)
                                try:
                                    mrr_result.append([i[0],self.imdb_person_dict[candidate[imputation]],Confidence(DPScores_OP[0])])
                                except:
                                    mrr_result.append([i[0],candidate[imputation],Confidence(DPScores_OP[0])])
                            else:
                                mrr_result.append([i[0], '', 0])
                        else:
                            mrr_result.append([i[0], '', 0])
                return mrr_1, mrr_result

            imdb_split_result_right_ = imdb_split.apply(mrr_test_movie, axis=1, result_type='expand')

            results = imdb_split_result_right_[1].values
            results = [i[0][1] if len(i) > 0 else "" for i in results]
            # A_result = []
            # for i in imdb_split_result_right_.iloc[:,0].values:
            #     A_result.extend(i)
            # print(np.mean(A_result), len(A_result))
            return results

        results = None
        if self.if_parallel is False:
            results = m_d_imputation(tuples_df)  # a list of values of str type

        return results

    def impute_Md_batch_dblp(self, tuples_df, missing_attribute, reverse_dict=False):
        dblp_embed_graph = self.dblp_embedding
        KG_IDs = list(self.dblp_t5_embedding_dict.keys())
        random.seed(1000)
        random.shuffle(KG_IDs)
        KG_size = len(KG_IDs)
        used_size = int(KG_size * self.mask_ratio)
        KG_IDs = KG_IDs[:used_size]
        for k in KG_IDs:
            dblp_embed_graph[self.dblp_t5_embedding_dict[k]] = np.zeros((1, dblp_embed_graph.shape[1]))

        def Confidence(array):
            a = np.array(array)
            vary = np.max(a) - np.min(a)
            a = a - np.min(a)
            score = np.max(a) / (np.sum(a) + 0.00000001)
            return score

        def TransIndex_dblp(row):
            temp = []
            for x, y in row[:9].iteritems():
                if (not pd.isnull(y)) and (y != '-'):
                    temp.extend(DBLPStringProcess_str(str(y).split('^^')))
                    if (x == 'vol') and len(y.split('^^')) == 1:
                        temp.append(13041)
                else:
                    temp.append(13041) if x != 'vol' else temp.extend([13041, 13041])
            vector = dblp_embed_graph[temp].reshape((2000))
            return vector

        def DBLPStringProcess_str(List):
            List_new = []
            for l in List:
                try:
                    List_new.append(self.dblp_t5_embedding_dict[l])
                except:
                    List_new.append(13041)
            return List_new

        def CandidateAwardDBLP_all(person, attribute):  ## querying candidate list for type/publish/year/page,return candidate list/label
            candidate_award = []
            candidate_person_list = list(set(self.dblp_DisAM_dict[person].split(',') + [person]))

            for id in candidate_person_list:
                if self.dblp_dict[id].__contains__(self.dblp_type_dict[attribute]):
                    candidate_award.extend(self.dblp_dict[id][self.dblp_type_dict[attribute]].split('||'))
            candidate_award_all = list(set(candidate_award))
            try:
                person_award_all = self.dblp_dict[person][self.dblp_type_dict[attribute]].split('||')
            except:
                person_award_all = []
            labels = np.zeros(len(candidate_award_all))
            for i in person_award_all:
                labels[candidate_award_all.index(i)] = 1
            return candidate_award_all, labels

        def DBLPEmbedLoad(List):
            TitleInfo = List
            if len(List) > 1:
                retrieve = itemgetter(*TitleInfo)(self.dblp_t5_embedding_dict)
                return dblp_embed_graph[list(retrieve)]
            else:
                return dblp_embed_graph[self.dblp_t5_embedding_dict[TitleInfo[0]]].reshape((-1, 200))

        def dblp_m_d(df_tuples):
            df = copy.deepcopy(df_tuples)
            length = 9
            df = df.reset_index(drop=True)
            df['Index'] = df.index
            df = df.fillna('-')
            dblp_split = df

            dblp_split_embed = df.iloc[:, :9].apply(TransIndex_dblp, axis=1, result_type='expand')
            dblp_split_embed = np.array(dblp_split_embed).reshape((-1, 10, 200))

            wiki_split_embed_result = np.zeros((len(dblp_split_embed), 1, 200))

            for i in range(np.int64(np.ceil(len(dblp_split_embed) / 5000))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(dblp_split_embed[i * 5000: min((i + 1) * 5000, len(dblp_split_embed))]).to(self.device)
                vector = vector.to(torch.float32)  # capable of inference with torch.float32
                result = self.model_load.forward(vector).detach()  # inference model, detach from grad()
                for j in range(i * 5000, min((i + 1) * 5000, len(dblp_split_embed)), 1):
                    wiki_split_embed_result[j] = self.avg_pool(result[j - i * 5000].reshape((1, 10, 200)))[0].cpu().numpy()
                torch.cuda.empty_cache()

            def mrr_test(A_):
                Index = A_['Index']

                col = pd.DataFrame(A_).T.columns
                id = A_['id']

                A_ = A_[~A_.isnull()]

                A_empty = A_[A_ == '-']
                A_fill = A_[A_ != '-']

                mrr_1 = []
                mrr_result = []
                mrr_2 = []

                for i in A_empty.iteritems():
                    if i[0] != missing_attribute:
                        continue
                    if i[0] in col[:length]:
                        if i[0].__contains__('author'):  # Haven't distinguished author_1 and author_2, TBD
                            attribute = 'author'
                        else:
                            attribute = i[0]
                        candidate, y_true = CandidateAwardDBLP_all(id, attribute)
                        if len(candidate) > 0:
                            candidate_embed = DBLPEmbedLoad(candidate)
                        else:
                            candidate_embed = dblp_embed_graph[13041].reshape((-1, 200))  # No Candidate List

                        candidate_person_list = self.dblp_DisAM_dict[id].split(',') + [id]
                        candidate_person_embed = DBLPEmbedLoad(candidate_person_list)

                        entity_embed = torch.from_numpy(wiki_split_embed_result[Index].reshape(1, 200))  # replace TransAttribute, get Transformer results
                        candidate_person_embed_torch = torch.from_numpy(candidate_person_embed)
                        Score = torch.argmax(torch.cosine_similarity(entity_embed, candidate_person_embed_torch)).numpy()
                        entity_embed = candidate_person_embed[Score]

                        if id == candidate_person_list[Score]:
                            mrr_2.append(1)
                        else:
                            mrr_2.append(0)

                        index = faiss.IndexFlatIP(200)
                        index.add(candidate_embed)
                        DPScores_OP, neighbors_OP = index.search(entity_embed.reshape((-1, 200)), candidate_embed.shape[0])

                        imputation = neighbors_OP[0][0]
                        if len(y_true) > 0:  # Have Candidate List
                            if np.sum(y_true == 1) > 0:  # Have True Value
                                if y_true[imputation] == 1:
                                    mrr_1.append(1)
                                else:
                                    mrr_1.append(0)
                            try:  # col = type/author_1/author_2
                                mrr_result.append([i[0], self.dblp_person_name_dict_reverse_generate[candidate[imputation]], Confidence(DPScores_OP[0])])
                            except:
                                mrr_result.append([i[0], candidate[imputation], Confidence(DPScores_OP[0])])
                        else:
                            mrr_result.append([i[0], '', 0])
                return mrr_1, mrr_2, mrr_result

            A_mrr_all = dblp_split.apply(mrr_test, axis=1, result_type='expand')

            results = A_mrr_all[2].values  # mrr_result
            results = [i[0][1] if len(i) > 0 else "" for i in results]

            return results

        return dblp_m_d(tuples_df)  # a list of values of str type

    def impute_Md_batch_college(self, tuples_df, missing_attribute):
        KG_IDs = list(self.college_embedding_dict.keys())
        college_embed = self.college_embed
        random.seed(1000)
        random.shuffle(KG_IDs)
        KG_size = len(KG_IDs)
        used_size = int(KG_size * self.mask_ratio)
        KG_IDs = KG_IDs[:used_size]
        for k in KG_IDs:
            college_embed[self.college_embedding_dict[k]] = np.zeros((1,college_embed.shape[1]))

        def TransIndex_college(row):
            temp = []
            row = row[:self.sequence_dim]
            for x, y in row.iteritems():
                if (not pd.isnull(y)) and (y != '-') and (y != 'nan'):
                    temp.append(self.college_embedding_dict[str(y).replace(' ', '_')])
                else:
                    temp.append(104792)
            vector = college_embed[temp].reshape((1600))
            return vector

        def CandidateAwardAll(person, attribute):
            candidate_award = []
            candidate_person_list = self.college_merge_dict[person] + [person]
            candidate_person_list = list(set(candidate_person_list))
            for i in candidate_person_list:
                candidate_award.append(self.college_info_all[i][attribute])
            candidate_award_all = list(set(candidate_award))
            person_award_all = [self.college_info_all[person][attribute]]

            labels = np.zeros(len(candidate_award_all))
            for i in person_award_all:
                labels[candidate_award_all.index(i)] = 1
            return candidate_award_all, labels

        def college_embed_load(List):
            if len(List) > 1:
                List = [str(i).replace(' ', '_') for i in List if not pd.isnull(i)]
                retrieve = itemgetter(*List)(self.college_embedding_dict)
                return college_embed[list(retrieve)]
            else:
                return college_embed[self.college_embedding_dict[List[0].replace(' ', '_')]].reshape((-1, 200))

        def Confidence(array):
            a = np.array(array)
            vary = np.max(a) - np.min(a)
            a = a - np.min(a)
            score = np.max(a) / (np.sum(a) + 0.00000001)
            return score

        def impute_Md(df_tuples):
            df = copy.deepcopy(df_tuples)
            college_split_result = df
            college_split_result = college_split_result[~college_split_result['id'].isnull()].reset_index(drop=True)
            college_split_result['id'] = college_split_result['id'].astype('float').astype('int')
            college_split_result = college_split_result.fillna('-')
            df = college_split_result
            df = df.reset_index(drop=True)
            df['Index'] = df.index
            df = df.replace("nan", np.nan)

            college_split_embed = df.apply(TransIndex_college, axis=1, result_type='expand')
            college_split_embed = np.array(college_split_embed).reshape((-1, self.sequence_dim, self.embedding_dim))

            college_split_embed_result = np.zeros((len(college_split_embed), 1, self.embedding_dim))

            for i in range(np.int64(np.ceil(len(college_split_embed) / 5000))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(college_split_embed[i * 5000: min((i + 1) * 5000, len(college_split_embed))]).to(self.device)
                vector = vector.to(torch.float32)  # capable of inference with torch.float32
                result = self.model_load.forward(vector).detach()  # inference model, detach from grad()
                for j in range(i * 5000, min((i + 1) * 5000, len(college_split_embed)), 1):
                    college_split_embed_result[j] = self.avg_pool(result[j - i * 5000].reshape((1, self.sequence_dim, self.embedding_dim)))[0].cpu().numpy()

            def mrr_test(A_):
                Index = A_['Index']
                # id = int(float(A_['id']))
                id = A_['id']
                A_ = A_[:8]
                col = pd.DataFrame(A_).T.columns

                A_ = A_[~A_.isnull()]

                A_empty = A_[A_ == '-']
                A_fill = A_[A_ != '-']

                mrr_1 = []
                mrr_result = []

                for i in A_empty.iteritems():
                    if i[0] != missing_attribute:
                        continue
                    if i[0] in col[:8]:
                        candidate, y_true = CandidateAwardAll(id, i[0])
                        candidate_embed = college_embed_load(candidate)

                        candidate_person_list = self.college_merge_dict[id] + [id]
                        candidate_person_list = list(set(candidate_person_list))
                        candidate_person_embed = college_embed_load(candidate_person_list)

                        entity_embed = torch.from_numpy(college_split_embed_result[Index].reshape(1, self.embedding_dim))  # replace TransAttribute, get Transformer results
                        candidate_person_embed_torch = torch.from_numpy(candidate_person_embed)
                        Score = torch.argmax(torch.cosine_similarity(entity_embed, candidate_person_embed_torch)).numpy()
                        entity_embed = candidate_person_embed[Score]

                        index = faiss.IndexFlatIP(self.embedding_dim)
                        index.add(candidate_embed)
                        DPScores_OP, neighbors_OP = index.search(entity_embed.reshape((-1, self.embedding_dim)), candidate_embed.shape[0])
                        imputation = neighbors_OP[0][0]
                        if len(y_true) > 0:  # Have Candidate List
                            if np.sum(y_true == 1) > 0:  # Have True Value
                                if y_true[imputation] == 1:
                                    mrr_1.append(1)
                                else:
                                    mrr_1.append(0)
                            mrr_result.append([i[0], candidate[imputation], Confidence(DPScores_OP[0])])
                        else:
                            mrr_result.append([i[0], '', 0])

                return mrr_1, mrr_result  # first: whether correctly impute the missing attribute values; second: attribute name, imputed value, and confidence

            A_mrr_all = df.apply(mrr_test, axis=1, result_type='expand')
            return A_mrr_all

        results = None
        if self.if_parallel is False:
            results = impute_Md(tuples_df)

        results = results[1].values  # mrr_result
        results = [i[0][1] if len(i) > 0 else "" for i in results]

        return results  # a list of values of str type
