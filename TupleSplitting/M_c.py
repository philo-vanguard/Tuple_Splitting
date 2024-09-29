from TupleSplitting.func import *
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from CorrelationModel.Attention_Classification import ClassificationPredictor
import torch
import torch.nn.functional as F
import random
pd.set_option('mode.chained_assignment', None)

all_data_dir = "/tmp/tuple_splitting/dict_model/"
predictor_delta = TabularPredictor.load(all_data_dir + "attention/autogluon/vary_delta/dblp_0.2/", require_version_match=False)


# Mc model
class CorrelationModel:
    def __init__(self):
        self.data_name = None
        self.if_parallel = False
        self.training_ratio = None
        self.mask_ratio = None
        self.Mc_type = None

        self.m_c_prediction_dblp = None
        self.m_c_prediction_imdb = None
        self.m_c_prediction_persons = None
        self.m_c_prediction_college = None

        # for persons
        self.predictor = None
        self.wikidata_embed = None
        self.wikidata_dic = None
        self.item_dict = None
        self.item_dict_reverse = None
        self.cuda_batch_size = None
        self.device = None
        self.m_c_model_load = None
        self.Label_0 = None
        self.Label_1 = None
        self.embedding = None
        self.sequence_dim = None
        self.wiki_t5 = None

        # for imdb
        self.imdb_embedding_dict = None
        self.imdb_embed = None
        self.imdb_info_all = None
        self.imdb_person_dict = None
        self.imdb_dict_reverse = None
        self.imdb_person_dict_reverse = None
        self.DisAM_movie_dict = None
        self.cuda_batch_size = None
        self.weight = None
        self.embedding_dim = None

        # for dblp
        self.dblp_embedding_dict = None
        self.dblp_embedding = None
        self.dblp_DisAM_dict = None
        self.dblp_dict = None
        self.dblp_person_name_dict_reverse = None
        self.dblp_t5_embedding_dict = None
        self.avg_pool = None
        self.model_dim = None
        self.dblp_type_dict = None

        # for college
        self.college_info_all = None
        self.college_merge_dict = None
        self.college_shuffle_list = None
        self.college_embedding_dict = None
        self.college_embed = None

    def load_model_Mc(self, data_name, cuda, if_parallel, training_ratio, mask_ratio, Mc_type):
        self.data_name = data_name
        self.training_ratio = training_ratio
        self.mask_ratio = mask_ratio
        self.Mc_type = Mc_type

        if self.data_name == "persons":
            if self.if_parallel is False and self.Mc_type == "graph":
                self.wikidata_embed = np.load(all_data_dir + 'dict/wikimedia/wikidata_embed_shrink.npy', mmap_mode='r')
                self.wikidata_dic = np.load(all_data_dir + 'dict/wikimedia/wikidata_dict_shrink_person.npy', allow_pickle=True).item()
                # self.item_dict = np.load(all_data_dir + 'dict/wikimedia/item_dict_shrink.npy', allow_pickle=True).item()
                self.item_dict_reverse = np.load(all_data_dir + 'dict/wikimedia/item_dict_reverse_all.npy', allow_pickle=True).item()
                # self.wiki_t5 = np.load(all_data_dir + 'dict/wikimedia/wiki_t5_encode_sentence_embed.npy')
                self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                self.cuda_batch_size = 5000
                self.model_dim = 2
                self.embedding_dim = 200
                self.sequence_dim = 10
                model_dir = all_data_dir + "models/wiki/M_c_5/"  # repair 5%
                if training_ratio == 1.0:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '100.ckpt').to(self.device)
                elif training_ratio == 0.2:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '20.ckpt').to(self.device)
                elif training_ratio == 0.4:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '40.ckpt').to(self.device)
                elif training_ratio == 0.6:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '60.ckpt').to(self.device)
                elif training_ratio == 0.8:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '80.ckpt').to(self.device)
                self.Label_0 = torch.repeat_interleave(torch.LongTensor([0]), self.sequence_dim)
                self.Label_1 = torch.repeat_interleave(torch.LongTensor([1]), self.sequence_dim)

        elif self.data_name == "imdb":
            if self.if_parallel is False and self.Mc_type == "graph":
                self.imdb_embedding_dict = np.load(all_data_dir + 'dict/IMDB/imdb_embedding_shrink_dict_add.npy', allow_pickle=True).item()
                self.imdb_embed = np.load(all_data_dir + 'dict/IMDB/imdb_embed_shrink_add.npy', mmap_mode='r')
                # self.imdb_embed_bert = np.load(all_data_dir + 'dict/IMDB/imdb_t5_encode.npy', mmap_mode='r')
                self.imdb_info_all = np.load(all_data_dir + 'dict/IMDB/imdb_info_all.npy', allow_pickle=True).item()
                self.imdb_dict_reverse = np.load(all_data_dir + 'dict/IMDB/imdb_dict_reverse.npy', allow_pickle=True).item()
                self.imdb_person_dict_reverse = np.load(all_data_dir + 'dict/IMDB/imdb_person_dict_reverse.npy', allow_pickle=True).item()
                # self.DisAM_movie_dict = np.load(all_data_dir + 'dict/IMDB/DisAM_movie_dict.npy', allow_pickle=True).item()
                self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                self.Label_0 = torch.repeat_interleave(torch.LongTensor([0]), 10)
                self.Label_1 = torch.repeat_interleave(torch.LongTensor([1]), 10)
                self.cuda_batch_size = 10000
                self.model_dim = 2
                self.embedding_dim = 200
                model_dir = all_data_dir + "models/imdb/M_c_5/"  # repair 5%
                if training_ratio == 1.0:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '100.ckpt').to(self.device)
                elif training_ratio == 0.2:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '20.ckpt').to(self.device)
                elif training_ratio == 0.4:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '40.ckpt').to(self.device)
                elif training_ratio == 0.6:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '60.ckpt').to(self.device)
                elif training_ratio == 0.8:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '80.ckpt').to(self.device)
                self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, self.embedding_dim))
                self.weight = torch.from_numpy(self.imdb_embed)
                self.embedding = torch.nn.Embedding.from_pretrained(self.weight, padding_idx=2486245)

        elif self.data_name == "dblp":
            if self.if_parallel is False and self.Mc_type == "graph":
                self.dblp_embedding_dict = np.load(all_data_dir + 'dict/DBLP/dblp_embedding_dict.npy', allow_pickle=True).item()
                self.dblp_embedding = np.load(all_data_dir + 'dict/DBLP/dblp_embedding.npy')
                self.dblp_t5_embedding_dict = np.load(all_data_dir + 'dict/DBLP/dblp_t5_embedding_dict_str.npy', allow_pickle=True).item()
                self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                self.cuda_batch_size = 5000
                self.model_dim = 2
                self.embedding_dim = 200
                self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, self.embedding_dim))
                self.Label_0 = torch.repeat_interleave(torch.LongTensor([0]), 10)
                self.Label_1 = torch.repeat_interleave(torch.LongTensor([1]), 10)
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
                model_dir = all_data_dir + "models/DBLP/M_c_repair_limit/"
                if training_ratio == 1.0:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '100.ckpt').to(self.device)
                elif training_ratio == 0.2:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '20.ckpt').to(self.device)
                elif training_ratio == 0.4:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '40.ckpt').to(self.device)
                elif training_ratio == 0.6:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '60.ckpt').to(self.device)
                elif training_ratio == 0.8:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '80.ckpt').to(self.device)

        elif self.data_name == "college":
            if self.if_parallel is False and self.Mc_type == "graph":
                # self.college_info_all = np.load(all_data_dir + 'dict/college/college_info_all.npy', allow_pickle=True).item()
                # self.college_merge_dict = np.load(all_data_dir + 'dict/college/college_merge_dict.npy', allow_pickle=True).item()
                # self.college_shuffle_list = np.load(all_data_dir + 'dict/college/college_shuffle_list.npy')
                self.college_embedding_dict = np.load(all_data_dir + 'dict/college/college_embedding_dict.npy', allow_pickle=True).item()
                self.college_embed = np.load(all_data_dir + 'dict/college/college_embed.npy')
                self.sequence_dim = 8
                self.cuda_batch_size = 10000
                self.Label_0 = torch.repeat_interleave(torch.LongTensor([0]), self.sequence_dim)
                self.Label_1 = torch.repeat_interleave(torch.LongTensor([1]), self.sequence_dim)
                self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
                self.weight = torch.from_numpy(self.college_embed)
                self.embedding = torch.nn.Embedding.from_pretrained(self.weight, padding_idx=104792)
                self.model_dim = 2
                self.embedding_dim = 200
                model_dir = all_data_dir + "models/college/M_c/"
                if training_ratio == 1.0:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '100.ckpt').to(self.device)
                elif training_ratio == 0.2:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '20.ckpt').to(self.device)
                elif training_ratio == 0.4:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '40.ckpt').to(self.device)
                elif training_ratio == 0.6:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '60.ckpt').to(self.device)
                elif training_ratio == 0.8:
                    self.m_c_model_load = ClassificationPredictor.load_from_checkpoint(model_dir + '80.ckpt').to(self.device)

    def predictMcScore_new(self, t_barA_B):
        if self.data_name == "persons":
            return self.predictMcScore_persons(t_barA_B)
        elif self.data_name == "imdb":
            return self.predictMcScore_imdb(t_barA_B, reverse_dict=True)
        elif self.data_name == "dblp":
            return self.predictMcScore_dblp(t_barA_B, reverse_dict=True)
        elif self.data_name == "college":
            return self.predictMcScore_college(t_barA_B)

    def predictMcScore_persons(self, t_barA_B):
        results = None
        if self.if_parallel is False and self.Mc_type == 'graph':
            def TransAttribute_vector(row):
                temp = []
                for x, y in row.iteritems():
                    if (not pd.isnull(y)) and (y != '-'):
                        temp.append(self.wikidata_dic[int(y)])
                    else:
                        temp.append(411211)
                return temp

            def m_c_prediction(df):  # Batch Prediction
                df = df.reset_index(drop=True)
                df['Index'] = df.index
                wiki_split_embed = df.iloc[:, :10].apply(TransAttribute_vector, axis=1, result_type='expand')
                ref = np.array(wiki_split_embed).astype(int)
                if self.Mc_type== 'graph':
                    wiki_split_embed = self.wikidata_embed[ref]

                wiki_split_embed_result = np.zeros((len(wiki_split_embed), self.sequence_dim, self.model_dim))  # store the Transformer output after pooling

                for i in range(np.int64(np.ceil(len(wiki_split_embed) / self.cuda_batch_size))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                    vector = torch.from_numpy(wiki_split_embed[i * self.cuda_batch_size: min((i + 1) * self.cuda_batch_size, len(wiki_split_embed))]).to(self.device)
                    vector = vector.to(torch.float32).to(self.device)  # capable of inference with torch.float32
                    result = self.m_c_model_load.forward(vector).detach()  # inference model, detach from grad()

                    for j in range(i * self.cuda_batch_size, min((i + 1) * self.cuda_batch_size, len(wiki_split_embed)), 1):
                        wiki_split_embed_result[j] = result[j - i * self.cuda_batch_size].reshape((1, self.sequence_dim, self.model_dim))[0].cpu().numpy()
                    torch.cuda.empty_cache()

                def m_c_cross_entropy(row):
                    Index = int(row['Index'])
                    result = wiki_split_embed_result[Index].reshape((1, self.sequence_dim, self.model_dim))
                    result = torch.from_numpy(result)

                    Score_0 = F.cross_entropy(result[0], self.Label_0).numpy()  # simulate Cross Entropy Loss, should revise!
                    # Score_1 = F.cross_entropy(result[0], Label_1.).numpy()
                    # if(Score_0<Score_1):
                    #     output = 0
                    # else:
                    #     output = 1
                    # return output,Score_0
                    return Score_0.astype(np.float32)

                Transformer_Result = df.apply(m_c_cross_entropy, axis=1, result_type='expand')
                return Transformer_Result

            col_seq = ['given name', 'family name', 'place of birth', 'place of death', 'gender', 'country', 'achieve', 'occupation', 'educated at', 'member of']

            if len(t_barA_B.shape) == 1:
                test_all = t_barA_B.to_frame().transpose()
            else:
                test_all = copy.deepcopy(t_barA_B)

            old_columns = test_all.columns.tolist()
            for index, value in enumerate(old_columns):
                old_columns[index] = value.replace("_", " ")
            test_all.columns = old_columns
            if "id" in test_all.columns:
                test_all.drop(columns=["id"], inplace=True)
            test_all = test_all.reindex(columns=col_seq, fill_value=np.nan)
            for c in col_seq:
                test_all[c] = test_all[c].map(self.item_dict_reverse[c])

            results = m_c_prediction(test_all).to_list()

        return results  # a list with number of float type

    # If Input is String, not id, use LimitMap, else skip LimitMap
    def predictMcScore_imdb(self, t_barA_B, reverse_dict=False):
        def TransIndex_imdb_vector(row):
            temp = []
            for x, y in row.iteritems():
                if (not pd.isnull(y)) and (y != '-'):
                    if not isinstance(y, float):
                        try:
                            temp.append(self.imdb_embedding_dict[y.strip().replace(' ', '_')])
                        except:
                            temp.append(2486245)
                    else:
                        temp.append(self.imdb_embedding_dict[str(int(y))])
                else:
                    temp.append(2486245)
            return temp

        def LimitMap(row):
            col_limit = pd.DataFrame(row).T.columns
            if 'title' in col_limit:
                title = row['title']
            elif 'primaryTitle' in col_limit:
                title = row['primaryTitle']
            elif 'tconst' in col_limit:
                title = self.imdb_info_all[row['tconst']]['primaryTitle']
                row['title'] = title

            for x, y in row.iteritems():
                if (x in ['actor', 'actress', 'writer', 'producer', 'director']) and not pd.isnull(y):
                    try:
                        row[x] = self.imdb_dict_reverse[title][y]
                    except:
                        row[x] = self.imdb_person_dict_reverse[y]
                elif x == 'genres' and not pd.isnull(y):
                    row[x] = np.random.choice(y.split(','),1)[0]
                elif x in ['startYear', 'runtimeMinutes'] and not pd.isnull(y):
                    row[x] = str(int(y))
            return row

        def m_c_prediction(df):  # Batch Prediction
            df = df.reset_index(drop=True)
            df['Index'] = df.index

            wiki_split_embed = df.iloc[:, :10].apply(TransIndex_imdb_vector, axis=1, result_type='expand')

            ref = np.array(wiki_split_embed).astype(int)
            if self.Mc_type == 'graph':
                wiki_split_embed = self.imdb_embed[ref]

            wiki_split_embed_result = np.zeros((len(wiki_split_embed), 10, self.model_dim))  # store the Transformer output after pooling
            for i in range(np.int64(np.ceil(len(wiki_split_embed) / self.cuda_batch_size))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(wiki_split_embed[i * self.cuda_batch_size: min((i + 1) * self.cuda_batch_size, len(wiki_split_embed))]).to(self.device)
                vector = vector.to(torch.float32)  # capable of inference with torch.float32
                result = self.m_c_model_load.forward(vector).detach()  # inference model, detach from grad()
                for j in range(i * self.cuda_batch_size, min((i + 1) * self.cuda_batch_size, len(wiki_split_embed)), 1):
                    wiki_split_embed_result[j] = result[j - i * self.cuda_batch_size].reshape((1, 10, self.model_dim))[0].cpu().numpy()
                torch.cuda.empty_cache()

            def m_c_cross_entropy(row):
                Index = int(row['Index'])

                result = wiki_split_embed_result[Index].reshape((1, 10, self.model_dim))
                result = torch.from_numpy(result)

                Score_0 = F.cross_entropy(result[0], self.Label_0).numpy()  # simulate Cross Entropy Loss, should revise!
                # Score_1 = F.cross_entropy(result[0], self.Label_1).numpy()
                # if (Score_0 < Score_1):
                #     output = 0
                # else:
                #     output = 1
                # return output, Score_0, Score_1
                return Score_0.astype(np.float32)

            Transformer_Result = df.apply(m_c_cross_entropy, axis=1, result_type='expand')
            return Transformer_Result

        if len(t_barA_B.shape) == 1:
            test_all = t_barA_B.to_frame().transpose()
        else:
            test_all = copy.deepcopy(t_barA_B)

        results = None
        if self.Mc_type == 'graph' and self.if_parallel is False:
            if reverse_dict:
                test_all = test_all.apply(LimitMap, axis=1)
                test_all = test_all.rename(columns={'primaryTitle': 'title'}).reindex(
                    columns=['titleType', 'title', 'startYear', 'runtimeMinutes', 'actor', 'actress',
                            'director', 'producer', 'writer', 'genres'], fill_value=np.nan
                )
            results = m_c_prediction(test_all)
            results = [i.tolist() for i in results]

        return results  # a list with number of float type

    def predictMcScore_dblp(self, t_barA_B, reverse_dict=False):
        def m_c_prediction(df):  # Batch Prediction
            if self.Mc_type == 'graph':
                dblp_embed_graph = self.dblp_embedding
            KG_IDs = list(self.dblp_t5_embedding_dict.keys())
            random.seed(1000)
            random.shuffle(KG_IDs)
            KG_size = len(KG_IDs)
            used_size = int(KG_size * self.mask_ratio)

            KG_IDs = KG_IDs[:used_size]
            for k in KG_IDs:
                dblp_embed_graph[self.dblp_t5_embedding_dict[k]] = np.zeros((1,dblp_embed_graph.shape[1]))

            def DBLPStringProcess_str(List):
                List_new = []
                for l in List:
                    if(l[-2:]=='.0'):
                        l = l[:-2]
                        
                    try:
                        List_new.append(self.dblp_t5_embedding_dict[l])
                    except:
                        List_new.append(self.dblp_embedding_dict[l])
                return List_new

            def TransIndex_dblp(row):
                temp = []
                for x, y in row[:9].iteritems():
                    if (not pd.isnull(y)) and (y != '-'):
                        temp.extend(DBLPStringProcess_str(str(y).split('^^')))
                        if (x == 'vol') and len(y.split('^^')) == 1:
                            temp.append(13041)
                    else:
                        temp.append(13041) if x != 'vol' else temp.extend([13041, 13041])
                vector = dblp_embed_graph[temp].reshape((self.embedding_dim * 10))
                return vector
            df = df.reset_index(drop=True)
            df = df.reindex(columns=['type', 'title', 'ref', 'author_1', 'author_2', 'publish', 'vol', 'year', 'page'], fill_value=np.nan)
            df['Index'] = df.index
            dblp_split_embed = df.iloc[:, :9].apply(TransIndex_dblp, axis=1, result_type='expand')
            dblp_split_embed = np.array(dblp_split_embed).reshape((-1, 10, self.embedding_dim))
            wiki_split_embed_result = np.zeros((len(dblp_split_embed), 10, self.model_dim))  # store the Transformer output after pooling
            for i in range(np.int64(np.ceil(len(dblp_split_embed) / self.cuda_batch_size))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(dblp_split_embed[i * self.cuda_batch_size: min((i + 1) * self.cuda_batch_size, len(dblp_split_embed))]).to(self.device)
                vector = vector.to(torch.float32)  # capable of inference with torch.float32
                result = self.m_c_model_load.forward(vector).detach()  # inference model, detach from grad()
                for j in range(i * self.cuda_batch_size, min((i + 1) * self.cuda_batch_size, len(dblp_split_embed)), 1):
                    wiki_split_embed_result[j] = result[j - i * self.cuda_batch_size].reshape((1, 10, self.model_dim))[0].cpu().numpy()
                torch.cuda.empty_cache()

            Transformer_Result = predictor_delta.predict_proba(pd.DataFrame(wiki_split_embed_result.reshape((-1, 20))))[1].astype(np.float32)
            return Transformer_Result

        if len(t_barA_B.shape) == 1:
            test_all = t_barA_B.to_frame().transpose()
        else:
            test_all = copy.deepcopy(t_barA_B)

        results = None
        if self.if_parallel is False and self.Mc_type == 'graph':
            T_result = m_c_prediction(test_all)
            results = T_result.to_list()

        return results  # a list with number of float type

    def predictMcScore_college(self, t_barA_B):
        def m_c_prediction(df):  # Batch Prediction
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
            df = df.reset_index(drop=True)
            df = df.reindex(columns=['INSTNM', 'ADDR', 'CITY', 'STABBR', 'ZIP', 'COUNTYNM', 'LONGITUD', 'LATITUDE'], fill_value=np.nan)
            df['Index'] = df.index
            college_split_embed = df.iloc[:, :8].apply(TransIndex_college, axis=1, result_type='expand')
            college_split_embed = np.array(college_split_embed).reshape((-1, self.sequence_dim, self.embedding_dim))

            wiki_split_embed_result = np.zeros((len(college_split_embed), self.sequence_dim, self.model_dim))  # store the Transformer output after pooling

            for i in range(np.int64(np.ceil(len(college_split_embed) / self.cuda_batch_size))):  # Each batch contains 10000 tuples with 10*200 size embedding, est. 22GiB graphic memory usage
                vector = torch.from_numpy(college_split_embed[i * self.cuda_batch_size: min((i + 1) * self.cuda_batch_size, len(college_split_embed))]).to(self.device)
                vector = vector.to(torch.float32).to(self.device)  # capable of inference with torch.float32
                result = self.m_c_model_load.forward(vector).detach()  # inference model, detach from grad()

                for j in range(i * self.cuda_batch_size, min((i + 1) * self.cuda_batch_size, len(college_split_embed)), 1):
                    wiki_split_embed_result[j] = result[j - i * self.cuda_batch_size].reshape((1, self.sequence_dim, self.model_dim))[0].cpu().numpy()

                torch.cuda.empty_cache()

            def m_c_cross_entropy(row):
                Index = int(row['Index'])
                test = row[:10]  # the first 10 columns should be exactly 'given name', 'family name', 'place of birth', 'place of death','gender', 'country', 'achieve', 'occupation', 'educated at','member of'

                result = wiki_split_embed_result[Index].reshape((1, 8, self.model_dim))
                result = torch.from_numpy(result)

                Score_0 = F.cross_entropy(result[0], self.Label_0).numpy()  # simulate Cross Entropy Loss, should revise!
                # Score_1 = F.cross_entropy(result[0], self.Label_1).numpy()
                # if (Score_0 < Score_1):
                #     output = 0
                # else:
                #     output = 1
                # return output, Score_0, Score_1
                return Score_0.astype(np.float32)

            Transformer_Result = df.apply(m_c_cross_entropy, axis=1, result_type='expand')

            return Transformer_Result

        if len(t_barA_B.shape) == 1:
            test_all = t_barA_B.to_frame().transpose()
        else:
            test_all = copy.deepcopy(t_barA_B)

        results = None
        if self.if_parallel is False and self.Mc_type == 'graph':
            results = m_c_prediction(test_all)
            results = [i.tolist() for i in results]
        return results  # a list with number of float type
