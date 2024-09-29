import pandas as pd
import numpy as np
import random
from collections import Counter
import copy

all_data_dir = "/tmp/tuple_splitting/dict_model/"

class HER:
    def __init__(self):
        self.data_name = None
        self.varyKG = None
        self.vary_ratio = None
        self.default_ratio = None

        self.value2id = None

        # for persons
        self.DisAM_dict = None
        self.DisAM_id_dict = None
        self.person_info_all = None
        self.item_dict = None

        # for imdb
        self.imdb_info_all = None
        self.imdb_person_dict = None
        self.imdb_DisAM_movie_dict = None

        # for dblp
        self.dblp_dict = None
        self.dblp_DisAM_dict = None     
        # self.dblp_person_name_dict_reverse_generate = None

        # for college
        self.college_info_all = None

    def load_dict_for_HER(self, data_name, varyKG=False, vary_ratio=1.0, default_ratio=1.0):
        self.data_name = data_name
        self.varyKG = varyKG
        self.vary_ratio = vary_ratio
        self.default_ratio = default_ratio

        if self.data_name == "persons":
            self.item_dict = np.load(all_data_dir + 'dict/wikimedia/item_dict_shrink.npy', allow_pickle=True).item()
            self.DisAM_dict = np.load(all_data_dir + 'dict/wikimedia/DisAM_dict.npy', allow_pickle=True).item()
            self.DisAM_id_dict = np.load(all_data_dir + 'dict/wikimedia/DisAM_id_dict.npy', allow_pickle=True).item()
            self.person_info_all = np.load(all_data_dir + 'dict/wikimedia/person_info_all_full.npy', allow_pickle=True).item()
            self.value2id = np.load(all_data_dir + "dict/wikimedia/wiki_all_value_mapping.npy", allow_pickle=True).item()

        elif data_name == "imdb":
            self.imdb_info_all = np.load(all_data_dir + 'dict/IMDB/imdb_info_all.npy', allow_pickle=True).item()
            self.imdb_person_dict = np.load(all_data_dir + 'dict/IMDB/imdb_person_dict.npy', allow_pickle=True).item()
            self.imdb_DisAM_movie_dict = np.load(all_data_dir + 'dict/IMDB/DisAM_movie_dict.npy', allow_pickle=True).item()
            self.value2id = np.load(all_data_dir + "dict/IMDB/imdb_all_value_mapping_limit_seperate.npy", allow_pickle=True).item()

        elif data_name == "dblp":
            self.dblp_dict = np.load(all_data_dir + 'dict/DBLP/dblp_dict_t5.npy', allow_pickle=True).item()
            self.dblp_DisAM_dict = np.load(all_data_dir + 'dict/DBLP/dblp_DisAM_dict.npy', allow_pickle=True).item()
            self.value2id = np.load(all_data_dir + "dict/DBLP/dblp_all_value_mapping.npy", allow_pickle=True).item()

        elif data_name == "college":
            self.college_info_all = np.load(all_data_dir + 'dict/college/college_info_all.npy', allow_pickle=True).item()
            self.value2id = np.load(all_data_dir + 'dict/college/college_all_value_mapping.npy',allow_pickle=True).item()

    # revise DisAM_dict to the form: {id: set(id1, id2, ...)}; If varyKG or ratio != 1.0, we remove the ids of DisAM_dict that not exist in KG_IDs;
    # Also add the key into DisAM_dict[key], if key not exists in value of DisAM_dict[key]
    def revise_DisAM_dict(self, data_name):
        if data_name == "imdb":
            if (self.varyKG is True and self.vary_ratio < 1.0) or (self.varyKG is False and self.default_ratio < 1.0):
                KG_IDs = list(self.imdb_info_all.keys())
                random.seed(1000)
                random.shuffle(KG_IDs)
                KG_size = len(KG_IDs)
                if self.varyKG is True:
                    used_size = int(KG_size * self.vary_ratio)
                else:
                    used_size = int(KG_size * self.default_ratio)
                KG_IDs = KG_IDs[:used_size]
                KG_IDs_hash = {id: 1 for id in KG_IDs}

                new_imdb_DisAM_movie_dict = {}
                for k, v in self.imdb_DisAM_movie_dict.items():
                    new_imdb_DisAM_movie_dict[k] = set()
                    if k in KG_IDs_hash.keys():
                        new_imdb_DisAM_movie_dict[k].add(k)
                    for id in self.imdb_DisAM_movie_dict[k]:
                        if id in KG_IDs_hash.keys():
                            new_imdb_DisAM_movie_dict[k].add(id)
                self.imdb_DisAM_movie_dict = new_imdb_DisAM_movie_dict
            else:
                for k, v in self.imdb_DisAM_movie_dict.items():
                    self.imdb_DisAM_movie_dict[k] = set(v)
                    if k not in v:
                        self.imdb_DisAM_movie_dict[k].add(k)

        elif data_name == "persons":
            if (self.varyKG is True and self.vary_ratio < 1.0) or (self.varyKG is False and self.default_ratio < 1.0):
                KG_IDs = list(self.person_info_all.keys()) # int
                random.seed(1000)
                random.shuffle(KG_IDs)
                KG_size = len(KG_IDs)
                if self.varyKG is True:
                    used_size = int(KG_size * self.vary_ratio)
                else:
                    used_size = int(KG_size * self.default_ratio)
                KG_IDs = KG_IDs[:used_size]
                KG_IDs_hash = {id: 1 for id in KG_IDs}

                new_DisAM_dict_person = {}
                for k, v in self.DisAM_dict['person'].items():
                    new_DisAM_dict_person[k] = set()
                    if k in KG_IDs_hash.keys():
                        new_DisAM_dict_person[k].add(str(k))
                    for id in v.split(','):
                        if int(id) in KG_IDs_hash.keys():
                            new_DisAM_dict_person[k].add(id)
                self.DisAM_dict['person'] = new_DisAM_dict_person
            else:
                for k, v in self.DisAM_dict['person'].items():
                    self.DisAM_dict['person'][k] = set(v.split(','))
                    if str(k) not in v:
                        self.DisAM_dict['person'][k].add(str(k))

        elif data_name == "dblp":
            if (self.varyKG is True and self.vary_ratio < 1.0) or (self.varyKG is False and self.default_ratio < 1.0):
                KG_IDs = list(self.dblp_dict.keys())
                random.seed(1000)
                random.shuffle(KG_IDs)
                KG_size = len(KG_IDs)
                if self.varyKG is True:
                    used_size = int(KG_size * self.vary_ratio)
                else:
                    used_size = int(KG_size * self.default_ratio)
                KG_IDs = KG_IDs[:used_size]
                KG_IDs_hash = {id: 1 for id in KG_IDs}

                new_dblp_DisAM_dict = {}
                for k, v in self.dblp_DisAM_dict.items():
                    new_dblp_DisAM_dict[k] = set()
                    if k in KG_IDs_hash.keys():
                        new_dblp_DisAM_dict[k].add(k)
                    for id in v.split(','):
                        if id in KG_IDs_hash.keys():
                            new_dblp_DisAM_dict[k].add(id)
                self.dblp_DisAM_dict = new_dblp_DisAM_dict
            else:
                for k, v in self.dblp_DisAM_dict.items():
                    self.dblp_DisAM_dict[k] = set(v.split(','))
                    if k not in v:
                        self.dblp_DisAM_dict[k].add(k)

    def reference_KG_set_intersection(self, tuples):
        temp_df = copy.deepcopy(tuples)

        old_columns = temp_df.columns.tolist()
        if "UNITID" in old_columns:
            old_columns.remove("UNITID")
            temp_df = temp_df.drop(columns=["UNITID"])
        if "id" in old_columns:
            old_columns.remove("id")
            temp_df = temp_df.drop(columns=["id"])

        if self.data_name == "persons":
            for index, value in enumerate(old_columns):
                old_columns[index] = value.replace("_", " ")
            temp_df.columns = old_columns

        isMatch = pd.Series([False] * temp_df.shape[0])
        matched_entity_ids = pd.Series([None] * temp_df.shape[0])
        oneMatch = pd.Series([False] * temp_df.shape[0])
        multiMatch = pd.Series([False] * temp_df.shape[0])
        isMatch.index = temp_df.index
        matched_entity_ids.index = temp_df.index
        oneMatch.index = temp_df.index
        multiMatch.index = temp_df.index
        for index, tuple in temp_df.iterrows():
            tids = []
            for attr, value in tuple.items():
                if pd.isnull(value) or str(value) == "nan" or value is None:
                    continue
                tids = list(self.value2id[attr][str(value)])
                break

            for attr, value in tuple.items():
                if pd.isnull(value) or str(value) == "nan" or value is None:
                    continue
                if len(tids) == 0:
                    break
                tids = list(set(tids).intersection(set(self.value2id[attr][str(value)])))

            if len(tids) > 0:
                isMatch.loc[index] = True
            if len(tids) == 1:
                oneMatch.loc[index] = True
            if len(tids) > 1:
                multiMatch.loc[index] = True
            matched_entity_ids.loc[index] = tids

        return isMatch, matched_entity_ids, oneMatch, multiMatch

    def reference_KG_set_intersection_filter_candidates(self, tuples):
        temp_df = copy.deepcopy(tuples)
        if self.data_name == "imdb":
            def reference_KG_set_intersection_filter_candidates_imdb(row_input):
                imdb_col = ['title', 'actor', 'actress', 'director', 'producer', 'writer', 'runtimeMinutes', 'startYear', 'genres', 'titleType']
                row = row_input.copy()
                imdb_dict = self.value2id[row['id']]
                candidate_ids = set(self.imdb_DisAM_movie_dict[row['id']])
                for x, y in row[imdb_col].iteritems():
                    if (not pd.isnull(y)) and (y != '') and (y != 'nan'):
                        if x in ['startYear', 'runtimeMinutes']:
                            y = str(int(y))
                        try:
                            candidate_ids = candidate_ids.intersection(set(imdb_dict[x][y]))
                        except:
                            candidate_ids = set()
                        if len(candidate_ids) == 0:
                            break
                row['isMatch'] = True if len(candidate_ids) > 0 else False
                row['matched_entity_ids'] = ["'" + str(c) + "'" for c in list(candidate_ids)] if len(candidate_ids) > 0 else []
                row['oneMatch'] = True if len(candidate_ids) == 1 else False
                row['multiMatch'] = True if len(candidate_ids) > 1 else False
                return row[-4:]

            temp_df_output = temp_df.apply(reference_KG_set_intersection_filter_candidates_imdb, axis=1)
            isMatch = temp_df_output.iloc[:, 0]
            matched_entity_ids = temp_df_output.iloc[:, 1]
            oneMatch = temp_df_output.iloc[:, 2]
            multiMatch = temp_df_output.iloc[:, 3]
        elif self.data_name == "persons":
            def reference_KG_set_intersection_filter_candidates_persons(row_input):
                row = row_input.copy()
                candidate_ids = set([int(i) for i in self.DisAM_dict['person'][self.DisAM_id_dict[int(row['id'])]]])
                for x, y in row[:10].iteritems():
                    if (not pd.isnull(y)) and (y != '') and (y != 'nan'):
                        x = x.replace('_', ' ')
                        candidate_ids.intersection_update(self.value2id[x][y])
                row['isMatch'] = True if len(candidate_ids) > 0 else False
                row['matched_entity_ids'] = list(candidate_ids)
                row['oneMatch'] = True if len(candidate_ids) == 1 else False
                row['multiMatch'] = True if len(candidate_ids) > 1 else False
                return row[-4:]

            temp_df_output = temp_df.apply(reference_KG_set_intersection_filter_candidates_persons, axis=1)
            isMatch = temp_df_output.iloc[:, 0]
            matched_entity_ids = temp_df_output.iloc[:, 1]
            oneMatch = temp_df_output.iloc[:, 2]
            multiMatch = temp_df_output.iloc[:, 3]
        elif self.data_name == "dblp":
            def reference_KG_set_intersection_filter_candidates_dblp(row_input):
                row = row_input.copy()
                # candidate_ids = set([i for i in self.dblp_DisAM_dict[row['id']].split(',')])
                candidate_ids = self.dblp_DisAM_dict[row['id']]
                candidate_ids = candidate_ids.union([row['id']])
                for x, y in row[:9].iteritems():
                    if (not pd.isnull(y)) and (y != '') and (y != 'nan'):
                        if (x == 'year'):
                            y = str(int(y))
                        candidate_ids.intersection_update(self.value2id[x][y])
                row['isMatch'] = True if len(candidate_ids) > 0 else False
                row['matched_entity_ids'] = list(candidate_ids)
                row['oneMatch'] = True if len(candidate_ids) == 1 else False
                row['multiMatch'] = True if len(candidate_ids) > 1 else False
                return row[-4:]

            temp_df_output = temp_df.apply(reference_KG_set_intersection_filter_candidates_dblp, axis=1)
            isMatch = temp_df_output.iloc[:, 0]
            matched_entity_ids = temp_df_output.iloc[:, 1]
            oneMatch = temp_df_output.iloc[:, 2]
            multiMatch = temp_df_output.iloc[:, 3]
        else:
            old_columns = temp_df.columns.tolist()
            if self.data_name == "persons":
                for index, value in enumerate(old_columns):
                    old_columns[index] = value.replace("_", " ")
                temp_df.columns = old_columns

            candidate_IDs = pd.Series([None] * temp_df.shape[0])
            candidate_IDs.index = temp_df.index
            # if self.data_name == "imdb":
            #     distinct_ID_in_KG = temp_df['id'].values
            #     candidate_IDs.loc[:] = [list(self.imdb_DisAM_movie_dict[i]) for i in distinct_ID_in_KG]
            # elif self.data_name == "persons":
            #     distinct_ID_in_KG = temp_df['id'].astype(int).values
            #     DisAM_id_in_KG = [int(self.DisAM_id_dict[i]) for i in distinct_ID_in_KG]
            #     candidate_IDs.loc[:] = [list(self.DisAM_dict['person'][i]) for i in DisAM_id_in_KG]
            # elif self.data_name == "dblp":
            #     distinct_ID_in_KG = temp_df['id'].values
            #     candidate_IDs.loc[:] = [list(self.dblp_DisAM_dict[i]) for i in distinct_ID_in_KG]
            # elif self.data_name == "college":  # this data does not have DisAM_id_dict. Thus, all tuples in KG can be the candidates.
            KG_IDs = list(self.college_info_all.keys())
            if (self.varyKG is True and self.vary_ratio < 1.0) or (self.varyKG is False and self.default_ratio < 1.0):
                random.seed(1000)
                random.shuffle(KG_IDs)
                KG_size = len(KG_IDs)
                if self.varyKG is True:
                    used_size = int(KG_size * self.vary_ratio)
                else:
                    used_size = int(KG_size * self.default_ratio)
                KG_IDs = KG_IDs[:used_size]
            candidate_IDs.loc[:] = [KG_IDs] * temp_df.shape[0]

            if "UNITID" in old_columns:
                old_columns.remove("UNITID")
                temp_df = temp_df.drop(columns=["UNITID"])
            if "id" in old_columns:
                old_columns.remove("id")
                temp_df = temp_df.drop(columns=["id"])

            isMatch = pd.Series([False] * temp_df.shape[0])
            matched_entity_ids = pd.Series([None] * temp_df.shape[0])
            oneMatch = pd.Series([False] * temp_df.shape[0])
            multiMatch = pd.Series([False] * temp_df.shape[0])
            isMatch.index = temp_df.index
            matched_entity_ids.index = temp_df.index
            oneMatch.index = temp_df.index
            multiMatch.index = temp_df.index
            for index, tuple in temp_df.iterrows():
                candidates = candidate_IDs.loc[index]
                tids = []
                for attr, value in tuple.items():
                    if pd.isnull(value) or str(value) == "nan" or value is None or str(value) == "":
                        continue
                    tids = list(set(self.value2id[attr][str(value)]).intersection(set(candidates)))
                    break

                for attr, value in tuple.items():
                    if pd.isnull(value) or str(value) == "nan" or value is None:
                        continue
                    if len(tids) == 0:
                        break
                    tids = list(set(tids).intersection(set(self.value2id[attr][str(value)])))

                if len(tids) > 0:
                    isMatch.loc[index] = True
                if len(tids) == 1:
                    oneMatch.loc[index] = True
                if len(tids) > 1:
                    multiMatch.loc[index] = True
                matched_entity_ids.loc[index] = tids

        return isMatch, matched_entity_ids, oneMatch, multiMatch

    def reference_KG_list_count(self, tuples):
        temp_df = copy.deepcopy(tuples)

        old_columns = temp_df.columns.tolist()
        if "UNITID" in old_columns:
            old_columns.remove("UNITID")
            temp_df = temp_df.drop(columns=["UNITID"])
        if "id" in old_columns:
            old_columns.remove("id")
            temp_df = temp_df.drop(columns=["id"])

        if self.data_name == "persons":
            for index, value in enumerate(old_columns):
                old_columns[index] = value.replace("_", " ")
            temp_df.columns = old_columns

        isMatch = pd.Series([False] * temp_df.shape[0])
        matched_entity_ids = pd.Series([None] * temp_df.shape[0])
        oneMatch = pd.Series([False] * temp_df.shape[0])
        multiMatch = pd.Series([False] * temp_df.shape[0])
        isMatch.index = temp_df.index
        matched_entity_ids.index = temp_df.index
        oneMatch.index = temp_df.index
        multiMatch.index = temp_df.index
        for index, tuple in temp_df.iterrows():
            all_tids = []
            non_null_value_counts = 0
            matched = True
            for attr, value in tuple.items():
                if pd.isnull(value) or str(value) == "nan" or value is None:
                    continue
                non_null_value_counts = non_null_value_counts + 1
                tids = list(self.value2id[attr][str(value)])
                if len(tids) == 0:
                    matched = False
                    break
                all_tids = all_tids + tids

            if non_null_value_counts == 0:
                continue

            if matched is False:
                continue

            match_tids = []
            for tid, count in Counter(all_tids).items():
                if count != non_null_value_counts:
                    break
                match_tids.append(tid)

            if len(match_tids) == 0:
                continue

            isMatch.loc[index] = True
            if len(match_tids) == 1:
                oneMatch.loc[index] = True
            elif len(match_tids) > 1:
                multiMatch.loc[index] = True
            matched_entity_ids.loc[index] = match_tids

        return isMatch, matched_entity_ids, oneMatch, multiMatch

    def imputation_via_HER_batch(self, tuples, missing_attribute, impute_all_empty_values=False):
        if self.data_name == "persons":
            return self.imputation_via_HER_batch_persons(tuples, missing_attribute, impute_all_empty_values)
        elif self.data_name == "imdb":
            return self.imputation_via_HER_batch_imdb(tuples, missing_attribute, impute_all_empty_values)
        elif self.data_name == "dblp":
            return self.imputation_via_HER_batch_dblp(tuples, missing_attribute)
        elif self.data_name == "college":
            return self.imputation_via_HER_batch_college(tuples, missing_attribute)

    # the current version is to obtain value from KG by matching the source_ids
    def imputation_via_HER_persons(self, tuple, complete_attributes, missing_attribute):
        retrieved_value = None

        # if not self.is_apply_KG:
        #     return retrieved_value

        distinct_ID_in_KG = int(tuple['id'])
        DisAM_id_in_KG = int(self.DisAM_id_dict[distinct_ID_in_KG])
        candidate_IDs = self.DisAM_dict['person'][DisAM_id_in_KG].split(',')  # obtain the IDs with the same main attributes

        for person_ID in list(candidate_IDs):
            person = self.person_info_all[int(person_ID)]
            for att in complete_attributes:
                if att == 'id':
                    continue
                if person[att.replace("_", " ")] != tuple[att]:
                    candidate_IDs.remove(person_ID)
                    break

        if len(candidate_IDs) == 1:
            # retrieved_value_ID = person_info_all[distinct_ID_in_KG][missing_attribute.replace("_", " ")].split(",")[0]
            retrieved_value_ID = self.person_info_all[int(candidate_IDs[0])][missing_attribute.replace("_", " ")].split(",")[0]
            if retrieved_value_ID != "":
                retrieved_value = self.item_dict[int(retrieved_value_ID)]

        # ------------------------via foreigh key
        # ground_truth_tuple_candidates = self.KG[self.KG['source_item_id'] == distinct_ID_in_KG]
        # candidate_count = len(ground_truth_tuple_candidates)
        # for iter in range(candidate_count):
        #     each_tuple = ground_truth_tuple_candidates.iloc[iter]
        #     if (each_tuple['property_value'] == missing_attribute):
        #         if (not pd.isnull(each_tuple['target_value'])):
        #             retrieved_value = each_tuple['target_value']
        #             break
        return retrieved_value

    '''
    # with bugs
    def imputation_via_HER_batch_persons(self, tuples, missing_attribute):
        indices_not_None = tuples.loc[tuples["id"].notnull()].index.values
        distinct_ID_in_KG = tuples.loc[indices_not_None]['id'].astype(int).values

        DisAM_id_in_KG = [int(self.DisAM_id_dict[i]) for i in distinct_ID_in_KG]
        candidate_IDs = [self.DisAM_dict['person'][i].split(',') for i in DisAM_id_in_KG]  # obtain the IDs with the same main attributes

        for idx in range(len(candidate_IDs)):
            tuple = tuples.loc[indices_not_None[idx]]
            for person_ID in candidate_IDs[idx]:
                person = self.person_info_all[int(person_ID)]
                complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
                if "id" in complete_attributes:
                    complete_attributes.remove("id")
                for att in complete_attributes:
                    if person[att.replace("_", " ")] != tuple[att]:
                        candidate_IDs[idx].remove(person_ID)
                        break

        one_candidate_indices = [idx for idx in range(len(candidate_IDs)) if len(candidate_IDs[idx]) == 1]
        if len(one_candidate_indices) == 0:
            return False, None

        retrieved_value_ID = [self.person_info_all[int(candidate_IDs[index][0])][missing_attribute.replace("_", " ")].split(",")[0] for index in one_candidate_indices]
        remove_indices = [idx for idx in range(len(retrieved_value_ID)) if retrieved_value_ID[idx] == ""]
        fill_value_indices = np.array([indices_not_None[i] for i in one_candidate_indices if i not in remove_indices])
        retrieved_value = [self.item_dict[int(retrieved_value_ID[idx])] for idx in range(len(retrieved_value_ID)) if idx not in remove_indices]

        tuples.loc[fill_value_indices, missing_attribute] = np.array(retrieved_value)
        return True, tuples[missing_attribute]
    '''

    def imputation_via_HER_batch_persons(self, tuples, missing_attribute, impute_all_empty_values=False):
        # KG_IDs = list(self.person_info_all.keys())
        # if self.varyKG is True:
        #     random.seed(1000)
        #     random.shuffle(KG_IDs)
        #     KG_size = len(KG_IDs)
        #     used_size = int(KG_size * self.vary_ratio)
        #     KG_IDs = KG_IDs[:used_size]

        # print("tuples size: ", tuples.shape[0])
        indices_not_None = tuples.loc[tuples["id"].notnull()].index.values
        # tuples_not_None = tuples.loc[indices_not_None]
        # indices_not_impute = tuples_not_None.loc[tuples_not_None["id"].astype(str).str.contains("\|\|", na=False)].index.values
        # indices_not_None = np.array([i for i in indices_not_None if i not in indices_not_impute])
        distinct_ID_in_KG = tuples.loc[indices_not_None]['id'].astype(int).values

        DisAM_id_in_KG = [int(self.DisAM_id_dict[i]) for i in distinct_ID_in_KG]
        
        # candidate_IDs = [self.DisAM_dict['person'][i].split(',') for i in DisAM_id_in_KG]  # obtain the IDs with the same main attributes; in the form of [[],[]]
        # candidate_IDs = [list(map(int, self.DisAM_dict['person'][i].split(','))) for i in DisAM_id_in_KG]  # obtain the IDs with the same main attributes; in the form of [[],[]]
        candidate_IDs = [list(self.DisAM_dict['person'][i]) for i in DisAM_id_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]

        # print("tuple size = ", len(tuples))
        # print("missing_attribute = ", missing_attribute)
        # print("candidate_IDs size = ", len(candidate_IDs))
        # len_count_0 = 0
        # len_count_1 = 0
        # len_count_2 = 0

        # for idx in range(len(distinct_ID_in_KG)): # in case that id is not be in DisAM_dict[i]
        #     _id = distinct_ID_in_KG[idx]
        #     if _id not in candidate_IDs[idx]:
        #         candidate_IDs[idx].append(_id)
        
        # candidate_IDs = [[id for id in candidate_IDs[idx] if id in KG_IDs] for idx in range(len(candidate_IDs))] # keep the IDs in the available KGs (i.e., KG_IDs)

        for idx in range(len(candidate_IDs)):  # idx \in [0, len-1]
            tuple = tuples.loc[indices_not_None[idx]]
            # if len(candidate_IDs[idx]) == 0:
            #     len_count_0 += 1

            complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
            if "id" in complete_attributes:
                complete_attributes.remove("id")
            # tuple = tuples.loc[idx]
            # tuple = tuples.iloc[idx]
            candidate_count = 0
            for person_ID in list(candidate_IDs[idx]):
                if candidate_count > 1: # more than one candidate founded
                    # len_count_2 += 1
                    break
                if int(person_ID) not in self.person_info_all.keys():
                    candidate_IDs[idx].remove(person_ID)
                    continue
                person = self.person_info_all[int(person_ID)]
                # complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
                # if "id" in complete_attributes:
                #     complete_attributes.remove("id")
                is_matched = True
                for att in complete_attributes:
                    # if person[att.replace("_", " ")] == '' or pd.isnull(person[att.replace("_", " ")]): # empty attribute value in KG dicts
                    #     candidate_IDs[idx].remove(person_ID)
                    #     is_matched = False
                    #     break
                    att_vals = [self.item_dict[int(id)] for id in person[att.replace("_", " ")].split(',') if id != '' and pd.notnull(id)] # obtain all values of this attribute of a tuple in KG
                    # if person[att.replace("_", " ")] != tuple[att]:
                    
                    # print("-------------------")
                    # print("tuple", tuple)
                    # print("att", att)
                    # print("att_vals", att_vals)
                    # print("att_vals[0]", att_vals[0])

                    if len(att_vals) >= 1:
                        if str(att_vals[0]) != '' and pd.notnull(str(att_vals[0])) and str(tuple[att]) not in att_vals:
                            candidate_IDs[idx].remove(person_ID)
                            is_matched = False
                            break
                if is_matched:
                    candidate_count += 1

            # if candidate_count == 0:
            #     len_count_0 += 1
            #     print("failed tuple id = ", tuple["id"])
            # elif candidate_count == 1:
            #     len_count_1 += 1
            # elif candidate_count > 1:
            #     len_count_2 += 1

        one_candidate_indices_idx = [idx for idx in range(len(candidate_IDs)) if len(candidate_IDs[idx]) == 1 and candidate_IDs[idx][0] != ""]
        one_candidate_indices = [indices_not_None[idx] for idx in one_candidate_indices_idx]
        # len_count_1_real = len(one_candidate_indices)
        # print("len_count_0", len_count_0)
        # print("len_count_1", len_count_1)
        # print("len_count_1_real", len_count_1_real)
        # print("len_count_2", len_count_2)
        if len(one_candidate_indices) == 0:
            return False, None, None

        # print("indices_not_None size: ", indices_not_None.shape[0])
        # print("distinct_ID_in_KG size: ", distinct_ID_in_KG.shape[0])
        # print("DisAM_id_in_KG size: ", len(DisAM_id_in_KG))
        # print("candidate_IDs size: ", len(candidate_IDs))
        # print("one_candidate_indices size: ", len(one_candidate_indices))

        # print("missing_attribute: ", missing_attribute.replace("_", " "))
        # print("int(candidate_IDs[index][0]): ", int(candidate_IDs[0][0]))
        # print("person_info_all[int(candidate_IDs[index][0])]: ", person_info_all[int(candidate_IDs[0][0])])
        # print("person_info_all[int(candidate_IDs[index][0])][missing_attribute]: ", person_info_all[int(candidate_IDs[0][0])][missing_attribute.replace("_", " ")])
        # print("person_info_all[int(candidate_IDs[index][0])][missing_attribute].split(",")[0]:", person_info_all[int(candidate_IDs[0][0])][missing_attribute.replace("_", " ")].split(",")[0])

        # retrieved_value_ID = [person_info_all[int(candidate_IDs[index][0])][missing_attribute.replace("_", " ")].split(",")[0] for index in one_candidate_indices]
        if impute_all_empty_values is False:
            all_retrieved_value_ID = [self.person_info_all[int(candidate_IDs[idx][0])][missing_attribute.replace("_", " ")].split(",")[0] if len(candidate_IDs[idx]) > 0 else "" for idx in range(len(candidate_IDs))]
            retrieved_one_match_value_ID = [int(all_retrieved_value_ID[idx]) for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]

            # print('--------------HER batch test--------------------')
            # print(retrieved_value_ID)
            # print("before retrieved_value_ID size: ", len(retrieved_one_match_value_ID))
            # remove_indices = [indices_not_None[idx] for idx in range(len(candidate_IDs)) if all_retrieved_value_ID[idx] == ""]
            # remove_indices = [idx for idx in range(len(candidate_IDs)) if all_retrieved_value_ID[idx] == ""]
            # remove_indices = [idx for idx in range(len(retrieved_value_ID)) if retrieved_value_ID[idx] == ""]
            # for rm_index in reversed(remove_indices):
            #     retrieved_value_ID.pop(rm_index)
            # fill_value_indices = np.array([indices_not_None[i] for i in retrieved_value_ID])
            # fill_value_indices = np.array([indices_not_None[i] for i in one_candidate_indices if i not in remove_indices])
            fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
            # fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and indices_not_None[idx] not in remove_indices])
            # print("before retrieved_value_ID: ", retrieved_value_ID)
            # while "" in retrieved_value_ID:
            #     retrieved_value_ID.remove("")
            # print("\nafter retrieved_value_ID: ", retrieved_value_ID)
            # retrieved_value = [item_dict[int(i)] for i in retrieved_value_ID]
            retrieved_value = [self.item_dict[idx] for idx in retrieved_one_match_value_ID]

            remove_indices = [index for index in range(len(retrieved_value)) if retrieved_value[index] == ""]
            if len(remove_indices) == len(retrieved_value):
                return False, None, None

            updated_indices = [fill_value_indices[i] for i in range(len(retrieved_value)) if i not in remove_indices]

            # print("after retrieved_value_ID size: ", len(retrieved_one_match_value_ID))
            # print("remove_indices size: ", len(remove_indices))
            # print("fill_value_indices size: ", len(fill_value_indices))
            # print("retrieved_value size: ", len(retrieved_value))

            # for test_idx in fill_value_indices:
            #     if len(str(test_idx).split(',')) > 1:
            #         print('------debug here----')
            #         print(test_idx)

            tuples.loc[fill_value_indices, missing_attribute] = np.array(retrieved_value)
            tuples.replace("", np.nan, inplace=True)
            return True, tuples[missing_attribute], np.array(updated_indices)
        else:
            updated_indices = []
            all_incomplete_attributes = [tuple.loc[pd.isnull(tuple)].index.values.tolist() for index, tuple in tuples.iterrows()]
            impute_success = False
            for idx in one_candidate_indices_idx:
                retrieved_one_match_value_ID = [self.person_info_all[int(candidate_IDs[idx][0])][missing_attr.replace("_", " ")].split(",")[0] for missing_attr in all_incomplete_attributes[idx]]
                remove_indices = [index for index in range(len(retrieved_one_match_value_ID)) if retrieved_one_match_value_ID[index] == ""]
                for rm_idx in reversed(remove_indices):
                    retrieved_one_match_value_ID.pop(rm_idx)
                    all_incomplete_attributes[idx].pop(rm_idx)

                retrieved_value = [self.item_dict[int(id)] for id in retrieved_one_match_value_ID]
                remove_indices = [index for index in range(len(retrieved_value)) if retrieved_value[index] == ""]
                for rm_idx in reversed(remove_indices):
                    retrieved_value.pop(rm_idx)
                    all_incomplete_attributes[idx].pop(rm_idx)

                if len(retrieved_value) > 0:
                    impute_success = True
                    tuples.loc[indices_not_None[idx], all_incomplete_attributes[idx]] = retrieved_value
                    updated_indices.append(indices_not_None[idx])

            tuples.replace("", np.nan, inplace=True)
            if impute_success is False:
                return False, None, None
            else:
                return True, tuples, np.array(updated_indices)

    def imputation_via_HER_batch_imdb(self, tuples, missing_attribute, impute_all_empty_values=False):
        # KG_IDs = list(self.imdb_info_all.keys())
        # if self.varyKG is True:
        #     random.seed(1000)
        #     random.shuffle(KG_IDs)
        #     KG_size = len(KG_IDs)
        #     used_size = int(KG_size * self.vary_ratio)
        #     KG_IDs = KG_IDs[:used_size]

        person_relatd_attributes_for_KG = ['actor', 'actress', 'director', 'producer', 'writer'] # attributes refer to imdb_person_dict
        attribute_may_involve_float = ['startYear', 'runtimeMinutes']
        # print("tuples size: ", tuples.shape[0])
        # indices_not_None = tuples.loc[tuples["id"].notnull()].index.values
        indices_not_None = tuples.loc[pd.notnull(tuples["id"])].index.values
        distinct_ID_in_KG = tuples.loc[indices_not_None]['id'].values

        # candidate_IDs = [[j.strip() for j in set(self.imdb_DisAM_movie_dict[i])] for i in distinct_ID_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]
        candidate_IDs = [list(self.imdb_DisAM_movie_dict[i]) for i in distinct_ID_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]
        # candidate_IDs = [[j.strip() for j in self.imdb_DisAM_movie_dict[i]] for i in distinct_ID_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]
        # candidate_IDs = [self.imdb_DisAM_movie_dict[i].split(',') for i in distinct_ID_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]

        # for idx in range(len(distinct_ID_in_KG)): # id may not be in imdb_DisAM_movie_dict[id]
        #     _id = distinct_ID_in_KG[idx]
        #     if _id not in candidate_IDs[idx]:
        #         candidate_IDs[idx].append(_id)

        # candidate_IDs = [[id for id in candidate_IDs[idx] if id in KG_IDs] for idx in range(len(candidate_IDs))] # keep the IDs in the available KGs (i.e., KG_IDs)

        candidate_size = len(candidate_IDs)
        for idx in range(candidate_size):  # idx \in [0, len-1]
            tuple = tuples.loc[indices_not_None[idx]]
            complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
            
            for att in list(complete_attributes):
                if "id" in complete_attributes:
                    complete_attributes.remove("id")
                elif tuple[att] is None or len(str(tuple[att])) == 0:
                    complete_attributes.remove(att)

            candidate_count = 0
            for movie_ID in list(candidate_IDs[idx]):
                if candidate_count > 1: # more than one candidate founded
                    break
                movie = self.imdb_info_all[movie_ID]
                # complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
                # if "id" in complete_attributes:
                #     complete_attributes.remove("id")
                is_matched = True
                for att in complete_attributes:
                    if movie[att] is None:
                        continue
                    elif pd.isnull(movie[att]) or len(movie[att]) == 0:
                        continue
                    att_vals = []
                    if att == 'title': # two (main and vice) titles split by ||
                        att_vals = movie[att].split('||')
                    elif att in person_relatd_attributes_for_KG:
                        if movie[att.replace("_", " ")] != '' and pd.notnull(movie[att.replace("_", " ")]):
                            # att_vals = [self.imdb_person_dict[id] for id in movie[att.replace("_", " ")].split(',') if id != '' and pd.notnull(id)] # obtain all values of this attribute of a movie in KG
                            att_vals = [self.imdb_person_dict[id] for id in movie[att.replace("_", " ")].split(',') if id != '' and pd.notnull(id) and id in self.imdb_person_dict.keys()] # obtain all values of this attribute of a movie in KG
                            # att_vals = np.concatenate(att_vals)
                    else: # only one attribute value
                        # if pd.notnull(movie[att]):
                        att_vals = str(movie[att]).split(',')
                    if att_vals is not None:
                        if att in attribute_may_involve_float:
                            if len(att_vals) > 0 and str(int(float(tuple[att]))) not in att_vals:
                                candidate_IDs[idx].remove(movie_ID)
                                is_matched = False
                                break
                        elif len(att_vals) > 0 and str(tuple[att]) not in att_vals:
                            candidate_IDs[idx].remove(movie_ID)
                            is_matched = False
                            break
                if is_matched:
                    candidate_count += 1

        one_candidate_indices_idx = [idx for idx in range(len(candidate_IDs)) if len(candidate_IDs[idx]) == 1 and candidate_IDs[idx][0] != ""]
        one_candidate_indices = [indices_not_None[idx] for idx in one_candidate_indices_idx]
        if len(one_candidate_indices) == 0:
            return False, None, None

        if impute_all_empty_values is False:
            fill_value_indices = []
            retrieved_value = []
            if missing_attribute == 'title':
                all_retrieved_value_ID = [self.imdb_info_all[candidate_IDs[idx][0]][missing_attribute.replace("_", " ")].split("||")[0] if len(candidate_IDs[idx]) > 0 else "" for idx in range(len(candidate_IDs))]
                retrieved_one_match_value_ID = [all_retrieved_value_ID[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]
                fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
                retrieved_value = [idx for idx in retrieved_one_match_value_ID] # all_retrieved_value_ID contains values after splitting
            elif missing_attribute in person_relatd_attributes_for_KG:
                all_retrieved_value_ID = [self.imdb_info_all[candidate_IDs[idx][0]][missing_attribute.replace("_", " ")].split(",")[0] if len(candidate_IDs[idx]) > 0 else "" for idx in range(len(candidate_IDs))]
                retrieved_one_match_value_ID = [all_retrieved_value_ID[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]
                fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
                retrieved_value = [self.imdb_person_dict[idx] if idx in self.imdb_person_dict.keys() else "" for idx in retrieved_one_match_value_ID]
            else:
                all_retrieved_value_ID = [self.imdb_info_all[candidate_IDs[idx][0]][missing_attribute.replace("_", " ")] if len(candidate_IDs[idx]) > 0 else "" for idx in range(len(candidate_IDs))]
                retrieved_one_match_value_ID = [all_retrieved_value_ID[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]
                fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
                retrieved_value = [idx.split(',')[0] for idx in retrieved_one_match_value_ID]
                # retrieved_value = [idx for idx in retrieved_one_match_value_ID]

            remove_indices = [index for index in range(len(retrieved_value)) if retrieved_value[index] == ""]
            if len(remove_indices) == len(retrieved_value):
                return False, None, None

            updated_indices = [fill_value_indices[i] for i in range(len(retrieved_value)) if i not in remove_indices]

            tuples.loc[fill_value_indices, missing_attribute] = np.array(retrieved_value)
            tuples.replace("", np.nan, inplace=True)
            return True, tuples[missing_attribute], np.array(updated_indices)
        else:
            all_incomplete_attributes = [tuple.loc[pd.isnull(tuple)].index.values.tolist() for index, tuple in tuples.iterrows()]
            impute_success = False
            updated_indices = []
            for idx in one_candidate_indices_idx:
                incomplete_attributes = all_incomplete_attributes[idx]
                retrieved_all_match_value_ID = [self.imdb_info_all[candidate_IDs[idx][0]][missing_attr.replace("_", " ")] for missing_attr in incomplete_attributes]
                retrieved_one_match_value_ID = [retrieved_all_match_value_ID[attr_id].split("||")[0] if incomplete_attributes[attr_id] == "title" else
                                                retrieved_all_match_value_ID[attr_id].split(",")[0] if incomplete_attributes[attr_id] in person_relatd_attributes_for_KG
                                                else retrieved_all_match_value_ID[attr_id]
                                                for attr_id in range(len(retrieved_all_match_value_ID))]
                remove_indices = [index for index in range(len(retrieved_one_match_value_ID)) if retrieved_one_match_value_ID[index] == ""]
                for rm_idx in reversed(remove_indices):
                    retrieved_one_match_value_ID.pop(rm_idx)
                    all_incomplete_attributes[idx].pop(rm_idx)

                incomplete_attributes = all_incomplete_attributes[idx]
                retrieved_value = [retrieved_one_match_value_ID[attr_id] if incomplete_attributes[attr_id] == "title" else
                                   (self.imdb_person_dict[retrieved_one_match_value_ID[attr_id]] if retrieved_one_match_value_ID[attr_id] in self.imdb_person_dict.keys() else "") if incomplete_attributes[attr_id] in person_relatd_attributes_for_KG
                                   else retrieved_one_match_value_ID[attr_id].split(',')[0]
                                   for attr_id in range(len(retrieved_one_match_value_ID))]
                remove_indices = [index for index in range(len(retrieved_value)) if retrieved_value[index] == ""]
                for rm_idx in reversed(remove_indices):
                    retrieved_value.pop(rm_idx)
                    all_incomplete_attributes[idx].pop(rm_idx)

                if len(retrieved_value) > 0:
                    impute_success = True
                    tuples.loc[indices_not_None[idx], all_incomplete_attributes[idx]] = retrieved_value
                    updated_indices.append(indices_not_None[idx])

            tuples.replace("", np.nan, inplace=True)
            if impute_success is False:
                return False, None, None
            else:
                return True, tuples, np.array(updated_indices)

    def imputation_via_HER_batch_dblp(self, tuples, missing_attribute):
        # KG_IDs = list(self.dblp_dict.keys())
        #
        # if self.varyKG is True:
        #     random.seed(1000)
        #     random.shuffle(KG_IDs)
        #     KG_size = len(KG_IDs)
        #     used_size = int(KG_size * self.vary_ratio)
        #     KG_IDs = KG_IDs[:used_size]

        dblp_map_dict = {
            'type': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
            'publish': 'https://dblp.org/rdf/schema#publishedIn',
            'year': 'https://dblp.org/rdf/schema#yearOfPublication',
            'page': 'https://dblp.org/rdf/schema#pagination',
            'author': 'https://dblp.org/rdf/schema#authoredBy',
            'vol': 'vol',
            'title': 'https://dblp.org/rdf/schema#title',
            'ref': 'http://www.w3.org/2000/01/rdf-schema#label',
        }

        indices_not_None = None
        if len(tuples.shape) == 1:
            if str(tuples["id"]) != "None" and str(tuples["id"]) != "nan":
                indices_not_None = tuples.index.values
        else:
            # indices_not_None = tuples.loc[tuples["id"].notnull()].index.values
            indices_not_None = tuples.loc[pd.notnull(tuples["id"])].index.values

        distinct_ID_in_KG = tuples.loc[indices_not_None]['id'].values

        # candidate_IDs = [self.dblp_DisAM_dict[i].split(',') for i in distinct_ID_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]
        candidate_IDs = [list(self.dblp_DisAM_dict[i]) for i in distinct_ID_in_KG] # obtain the IDs with the same main attributes; in the form of [[],[]]

        # for idx in range(len(distinct_ID_in_KG)): # id may not be in dblp_DisAM_dict[i]
        #     _id = distinct_ID_in_KG[idx]
        #     if _id not in candidate_IDs[idx]:
        #         candidate_IDs[idx].append(_id)

        # candidate_IDs = [[id for id in candidate_IDs[idx] if id in KG_IDs] for idx in range(len(candidate_IDs))] # keep the IDs in the available KGs (i.e., KG_IDs)

        for idx in range(len(candidate_IDs)):  # idx \in [0, len-1]
            tuple = tuples.loc[indices_not_None[idx]]
            candidate_count = 0

            for paper_ID in list(candidate_IDs[idx]):
                if candidate_count > 1: # more than one candidate founded
                    break
                paper = self.dblp_dict[paper_ID]
                complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
                if "id" in complete_attributes:
                    complete_attributes.remove("id")
                is_matched = True
                for att in complete_attributes:                    
                    att_vals = []
                    if att == "author_1":
                        if dblp_map_dict['author'] in paper.keys(): # first author
                            att_vals = paper[dblp_map_dict['author']].split('||')[0]
                    elif att == "author_2":
                        if dblp_map_dict['author'] in paper.keys(): # second author
                            att_vals = paper[dblp_map_dict['author']].split('||')
                            att_vals.pop(0)
                    elif att == "vol":
                        if dblp_map_dict[att] in paper.keys():
                            att_vals = paper[dblp_map_dict[att]]
                    elif dblp_map_dict[att] in paper.keys(): # other attributes
                        att_vals = paper[dblp_map_dict[att]].split('||')
                    
                    if str(tuple[att]).replace("^^", "||") not in att_vals:
                        candidate_IDs[idx].remove(paper_ID)
                        is_matched = False
                        break
                if is_matched:
                    candidate_count += 1

        one_candidate_indices = [indices_not_None[idx] for idx in range(len(candidate_IDs)) if len(candidate_IDs[idx]) == 1]
        if len(one_candidate_indices) == 0:
            return False, None, None
        
        fill_value_indices = []
        retrieved_value = []

        if missing_attribute == "author_1": # missing attribute is author_1
            all_retrieved_value_ID = [self.dblp_dict[candidate_IDs[idx][0]][dblp_map_dict['author']].split("||")[0] 
            if len(candidate_IDs[idx]) > 0 and dblp_map_dict['author'] in self.dblp_dict[candidate_IDs[idx][0]].keys()
            else "" for idx in range(len(candidate_IDs))]
            retrieved_one_match_value_ID = [all_retrieved_value_ID[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]
            fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
            retrieved_value = [idx for idx in retrieved_one_match_value_ID] 
        elif missing_attribute == "author_2": # missing attribute is author_2:
            all_retrieved_value_ID = [self.dblp_dict[candidate_IDs[idx][0]][dblp_map_dict['author']].split("||")[1] 
            if len(candidate_IDs[idx]) > 0 and dblp_map_dict['author'] in self.dblp_dict[candidate_IDs[idx][0]].keys() and len(self.dblp_dict[candidate_IDs[idx][0]][dblp_map_dict['author']].split("||")) > 1 
            else "" for idx in range(len(candidate_IDs))]
            retrieved_one_match_value_ID = [all_retrieved_value_ID[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]
            fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
            retrieved_value = [idx for idx in retrieved_one_match_value_ID] 
        else: # other attributes
            all_retrieved_value_ID = [self.dblp_dict[candidate_IDs[idx][0]][dblp_map_dict[missing_attribute.replace("_", " ")]].split("||")[0] 
            if len(candidate_IDs[idx]) > 0 and dblp_map_dict[missing_attribute] in self.dblp_dict[candidate_IDs[idx][0]].keys()
            else "" for idx in range(len(candidate_IDs))]
            retrieved_one_match_value_ID = [all_retrieved_value_ID[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""]
            fill_value_indices = np.array([indices_not_None[idx] for idx in range(len(candidate_IDs)) if indices_not_None[idx] in one_candidate_indices and all_retrieved_value_ID[idx] != ""])
            retrieved_value = [idx for idx in retrieved_one_match_value_ID]

        remove_indices = [index for index in range(len(retrieved_value)) if retrieved_value[index] == ""]
        if len(remove_indices) == len(retrieved_value):
            return False, None, None

        updated_indices = [fill_value_indices[i] for i in range(len(retrieved_value)) if i not in remove_indices]

        tuples.loc[fill_value_indices, missing_attribute] = np.array(retrieved_value)
        tuples.replace("", np.nan, inplace=True)
        return True, tuples[missing_attribute], np.array(updated_indices)

    def imputation_via_HER_batch_college(self, tuples, missing_attribute):
        KG_IDs = list(self.college_info_all.keys())

        if (self.varyKG is True and self.vary_ratio < 1.0) or (self.varyKG is False and self.default_ratio < 1.0):
            random.seed(1000)
            random.shuffle(KG_IDs)
            KG_size = len(KG_IDs)
            if self.varyKG is True:
                used_size = int(KG_size * self.vary_ratio)
            else:
                used_size = int(KG_size * self.default_ratio)
            KG_IDs = KG_IDs[:used_size]

        # print("tuples size: ", tuples.shape[0])
        indices_for_HER = [] # the tuple indices that can be filled via HER
        values_for_HER = [] # the filled values returned via HER ; indices_for_HER.size ： values_for_HER.size = 1 : 1
        for index_1, tuple in tuples.iterrows():
            if pd.isnull(tuple['id']): # HER cannot work
                continue
            complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
            if "id" in complete_attributes:
                complete_attributes.remove("id")
            # complete_attributes = []
            # for attribute in tuple.index.values.tolist():
            #     if "id" != attribute and missing_attribute != attribute and not pd.isnull(tuple[attribute]):
            #         complete_attributes.append(attribute)
            candidate_indices = []
            is_unique_identified = True
            # for index_2 in self.college_info_all.keys():
            # for index_2, tuple_in_KG in self.college_info_all.iterrows():
            for index_2 in KG_IDs: # only consider IDs in available KGs
                tuple_in_KG = self.college_info_all[index_2]
                is_match = True
                for attribute in complete_attributes:
                    if str(tuple_in_KG[attribute]) != str(tuple[attribute]) and pd.notnull(tuple_in_KG[attribute]):
                        is_match = False
                        break
                if is_match:
                    distinct_ID_in_KG = int(self.college_info_all[index_2]['UNITID'])
                    if distinct_ID_in_KG not in candidate_indices:
                        candidate_indices.append(distinct_ID_in_KG)
                    if len(candidate_indices) > 1:
                        is_unique_identified = False
                        break
            if len(candidate_indices) != 1:
                is_unique_identified = False
            if is_unique_identified: # unique identified
                val = self.college_info_all[int(candidate_indices[0])][missing_attribute]
                if pd.notnull(val):
                    indices_for_HER.append(index_1)
                    values_for_HER.append(val)
            elif len(candidate_indices) > 1:
                all_vals = [self.college_info_all[int(idx)][missing_attribute] for idx in candidate_indices if pd.notnull(self.college_info_all[int(idx)][missing_attribute])]
                val_frequency = Counter(all_vals)
                if pd.notnull(val_frequency.most_common(1)[0][0]) and int(val_frequency.most_common(1)[0][1]) == len(candidate_indices):
                    indices_for_HER.append(index_1)
                    values_for_HER.append(str(val_frequency.most_common(1)[0][0]))
        tuples.loc[np.array(indices_for_HER), missing_attribute] = np.array(values_for_HER)
        return True, tuples[missing_attribute], np.array(indices_for_HER)
