from TupleSplitting.rule import REELogic
from Imputation.func import *
from TupleSplitting.func import *
import random
import pandas as pd

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

random.seed(7654321)


class GS:
    # H holds the previously enforced valuations
    # SA holds the enforced valuations that may need to re-checked due to the update of A-attribute
    # A_star records the conflicting attribute in the last chase round
    def __init__(self):
        self.H = []
        self.SA = []
        self.A_star = None

    # call this function when an incomplete tuple is completed, for the usage of the completion of following tuples
    def re_init(self):
        self.H = []
        self.SA = []
        A_star = None

    def SA_initialize(self, missing_attributes):
        for idx in range(len(missing_attributes)):
            self.SA.append([])

    # add a valuation to H
    def add_H(self, h):
        if h not in self.H:
            self.H.append(h)

    # remove a valuation from H
    def remove_H(self, h):
        self.H.remove(h)

    # add a valuation h to SA, where h may need to be re-checked if the A-attribute updates
    def add_SA(self, h, a_pos_in_SA):
        self.SA[a_pos_in_SA].add(h)

    # remove a valuation from SA, when the deduced attribute B via h is validated
    def remove_SA(self, h, a_pos_in_SA):
        self.SA[a_pos_in_SA].remove(h)

    # call this function when chase terminates in an invalid result, due to the conflicts on A-attribute; A_star <-- A
    def update_A_star(self, attribute):
        self.A_star = attribute

    # this function is for debug purpose
    def print_info(self):
        print('---------the information of H is ----------\n')
        print(self.H)
        print('\n')
        print('---------the information of SA is ----------\n')
        print(self.SA)
        print('\n')
        print('---------the information of A_star is ----------\n')
        print(self.A_star)
        print('\n')


# class Imputation herits from class GS
class Imputation(GS):
    def __init__(self):
        super().__init__()
        super(GS, self).__init__()
        self.KG = None
        # self.M_d = Md()
        self.Gamma = None
        self.Gamma_attribute_value_with_highest_frequency = {}
        self.Sigma = []  # all REEs
        self.tuples = None
        self.useful_REEs = []  # the useful REEs for imputing a specific tuple
        self.operators = {'=': operator.eq, '<>': operator.ne, '>': operator.gt, '<': operator.lt, '>=': operator.ge,
                          '<=': operator.le}
        # self.gs = gs
        # self.is_apply_logic = True
        # self.is_apply_Md = True
        # self.is_apply_KG = True

        self.varyKG = False
        self.ratio = 1.0
        self.tuples_res = []

    def load_Sigma(self, Sigma_path, varyREE, vary_ratio, default_ratio):  # REE
        f = open(Sigma_path, "r", encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            ree = REELogic()
            ree.load_X_and_e(line)
            self.Sigma.append(ree)

        full_size = len(self.Sigma)
        if varyREE is True:
            use_size = int(full_size * vary_ratio)
            self.Sigma = self.Sigma[:use_size]
        else:
            use_size = int(full_size * default_ratio)
            self.Sigma = self.Sigma[:use_size]
        return self.Sigma

    def load_KG(self, KG_path, varyKG, ratio):
        # self.KG = pd.read_csv(KG_path, encoding='iso-8859-1', index_col=0)
        self.KG = pd.read_csv(KG_path, index_col=0)
        if varyKG is True:
            full_size = self.KG.shape[0]
            use_size = int(full_size * ratio)
            self.KG = self.KG.iloc[:use_size]

    def set_varyKG(self, varyKG, vary_ratio, default_ratio):
        self.varyKG = varyKG
        if self.varyKG is True:
            self.ratio = vary_ratio
        else:
            self.ratio = default_ratio

    def load_Gamma(self, Gamma_path, varyGT, vary_ratio, default_ratio):  # ground_truth
        # self.Gamma = pd.read_csv(Gamma_path, encoding='iso-8859-1', index_col=0)
        self.Gamma = pd.read_csv(Gamma_path, index_col=0)
        full_size = self.Gamma.shape[0]
        if varyGT is True:
            use_size = int(full_size * vary_ratio)
            self.Gamma = self.Gamma.iloc[:use_size]
        else:
            use_size = int(full_size * default_ratio)
            self.Gamma = self.Gamma.iloc[:use_size]
        return self.Gamma

    def load_Tuples(self, tuples_path):
        self.tuples = pd.read_csv(tuples_path)
        return self.tuples

    # update in the tuple unit; call this function when all attributes of a new tuple are validated
    def update_Gamma(self, new_validated_data):
        self.Gamma.loc[len(self.Gamma.index)] = new_validated_data

    # update Gamma by adding tuples
    def update_Gamma_by_tuples(self, new_validated_tuples):
        if len(new_validated_tuples.shape) == 1:  # series
            new_validated_tuples = new_validated_tuples.to_frame().transpose()  # transform the series to dataframe with one line

        # filter tuples, and only add those with id not exists in Gamma
        ids_in_GT = list(set(self.Gamma["id"].astype(str).values))
        addable_indices = new_validated_tuples[~new_validated_tuples["id"].astype(str).isin(ids_in_GT)].index.values
        self.Gamma = self.Gamma.append(new_validated_tuples.loc[addable_indices], ignore_index=True)

        return self.Gamma

    # remove the REEs from Sigma not for imputing missing attributes
    def initialize_useful_REEs(self, tuple, missing_attributes):
        self.useful_REEs = [i for i in self.Sigma]
        for ree in list(self.useful_REEs):
            # e_attribute = ree.get_e()[0].split('.')[1]
            e_attribute = ree.get_RHS().get_attr1()
            if e_attribute not in missing_attributes:
                self.useful_REEs.remove(ree)

    def initialize_data_structure(self, tuple):
        pass

    # calculate the confidence of filling t[A] as val via a valuation (h, varphi)
    def calculate_value_confidence(self, tuple):
        pass

    def get_info(self):
        for ree in self.useful_REEs:
            ree.print_info()

    def test_get_a_tuple(self, count):
        return self.Gamma.head(count)

    def imputation_via_Md(self, tuple, missing_attribute):  # impute tuple[missing_attribute] based on 1) the complete attributes of tuple and 2) the correlation model M_d
        # if not self.is_apply_Md:
        #     return None

        tuple[missing_attribute] = '-'
        # tuple_for_Md = pd.DataFrame(tuple).T.iloc[0]
        hit_result, result_and_confidence = impute_Md(tuple.iloc[0], self.DisAM_dict, self.DisAM_id_dict, self.item_dict, self.person_info_all, self.wikidata_embed, self.wikidata_dic, self.item_dict_reverse)
        # print('------------------Md test-----------------------')
        # print(hit_result)
        # print(result_and_confidence)
        if len(result_and_confidence) > 0:
            return result_and_confidence[0][1:3]  # return an imputed value; 0 --> attribute name; 1 --> attribute value; 2 --> condifence
        else:
            return [None, 0]

    def imputation_via_logic(self, tuple, missing_attribute, rule):
        result = None  # the imputed value (i.e., tuple[missing_attribute]) via this rule

        # if not self.is_apply_logic:
        #     return result

        # complete_attributes = []
        # for attribute in tuple.index:  # obtain the names of attributes with complete values
        #     if (not pd.isnull(tuple[attribute])):
        #         complete_attributes.append(attribute)

        tuple_variable_cnt = rule.get_tuple_variable_cnt()  # the number of tuples involved in this rule
        # # attributes_in_X_of_rule =  rule.get_distinct_attributes_in_X() # obtain the set of attribute names in X
        # preconditions = rule.get_X()  # obtain all predicates in X
        # consequence = rule.get_e()

        # # if (consequence[0].split('.')[1] != missing_attribute): # the consequence of rule is not related to missing_attribute
        # #     return result
        #
        # # if(not all(att in complete_attributes for att in attributes_in_X_of_rule)): # if complete_attributes cannot cover all attributes in preconditions X of rule, then this rule cannot be satisfied and enforced
        # #     return result

        # X_satisfied = True  # indicates whether the incomplete tuple satisfies the rule
        # tuple_in_Gamma_satisfied = True  # indicates whether the tuple in Gamma satisfies the rule
        # p_count = len(preconditions)  # the number of predicates in preconditions
        # matched_tuple_in_Gamma = ''

        if tuple_variable_cnt == 1:  # single-variable REE
            # for idx in range(p_count):
            #     precondition = preconditions[idx]
            #     X_attribute = precondition[0].split('.')[1]
            #     X_operator = precondition[1]
            #     X_value = precondition[2][1:len(precondition[2]) - 1]
            #     if not self.operators[X_operator](tuple[X_attribute], X_value):
            #         X_satisfied = False
            #         break
            violate, no_use = violateSingleRule(rule, tuple)
            constant = rule.get_RHS().get_constant()
            if violate is True and not pd.isnull(constant):  # satisfy X but not Y
                result = constant

        elif tuple_variable_cnt == 2:  # 't0' represents the tuple in Gamma, 't1' indicates the incomplete tuple
            # tuple_cnt_in_Gamma = self.Gamma.shape[0]
            # for idx in range(tuple_cnt_in_Gamma):
            #     tuple_in_Gamma = self.Gamma.iloc[idx]  # retrieve a tuple from ground truth Gamma
            #     tuple_in_Gamma_satisfied = True  # reset for another tuple from Gamma
            #     for idx_2 in range(p_count):
            #         precondition = preconditions[idx_2]
            #         len_1 = len(precondition[0].split('.'))  # len = 2 for the cases e.g., 't0.continent'
            #         len_2 = len(precondition[2].split('.'))  # len = 1 for the cases e.g., "'SA'"
            #         if len_2 == 1:  # X_value = "'SA'"
            #             attribute_tag = precondition[0].split('.')[0]
            #             X_attribute = precondition[0].split('.')[1]
            #             X_operator = precondition[1]
            #             X_value = precondition[2][1:len(precondition[2]) - 1]
            #             if attribute_tag == 't0':  # the tuple in Gamma, i.e., tuple_in_Gamma
            #                 if not X_operator(tuple_in_Gamma[X_attribute], X_value):
            #                     tuple_in_Gamma_satisfied = False
            #                     break
            #             elif attribute_tag == 't1':  # 't1' indicates the incomplete tuple, i.e., tuple
            #                 if not X_operator(tuple[X_attribute], X_value):
            #                     X_satisfied = False
            #                     break
            #         elif len_2 == 2:  # for predicates e.g., t0.municipality = t1.municipality
            #             X_attribute = precondition[0].split('.')[1]
            #             X_operator = precondition[1]
            #             if not self.operators[X_operator](tuple_in_Gamma[X_attribute], tuple[X_attribute]):
            #                 tuple_in_Gamma_satisfied = False
            #                 break
            #     if not X_satisfied:  # the incomplete tuple itself cannot satisfy this rule, no need to further check tuples from Gamma
            #         break
            #     elif tuple_in_Gamma_satisfied:
            #         matched_tuple_in_Gamma = tuple_in_Gamma
            #         break

            satisfy_cpredicates = satisfyConstantPredicatesInX(rule, tuple)

            # the incomplete tuple itself cannot satisfy this rule, no need to further check tuples from groundtruth Gamma
            if satisfy_cpredicates[0] is False and satisfy_cpredicates[1] is False:
                return result

            rhs = rule.get_RHS()
            if rhs.get_type() == "non-constant":
                for index, row in self.Gamma.iterrows():
                    violate1, violate2 = False, False
                    if satisfy_cpredicates[0] is True:  # tuple can be regarded as t0
                        violate1, no_use = violateBiVariableREE(rule, tuple, row)
                    if satisfy_cpredicates[1] is True and violate1 is False:  # tuple can be regard as t1
                        violate2, no_use = violateBiVariableREE(rule, row, tuple)
                    if (violate1 is True or violate2 is True) and not pd.isnull(row[missing_attribute]):  # satisfy X but not Y
                        result = row[missing_attribute]
                        break
            elif rhs.get_type() == "constant":
                for index, row in self.Gamma.iterrows():
                    violate = False
                    if rhs.get_index1() == 0 and satisfy_cpredicates[0] is True:  # tuple can be regard as t0
                        violate, no_use = violateBiVariableREE(rule, tuple, row)
                    elif rhs.get_index1() == 1 and satisfy_cpredicates[1] is True:  # tuple can be regard as t1
                        violate, no_use = violateBiVariableREE(rule, row, tuple)
                    if violate is True and not pd.isnull(rhs.get_constant()):  # satisfy X but not Y
                        result = rhs.get_constant()
                        break

        # if X_satisfied and tuple_in_Gamma_satisfied:
        #     if tuple_variable_cnt == 1:
        #         result = consequence[2][1:len(consequence[2]) - 1]
        #     elif tuple_variable_cnt == 2:
        #         result = matched_tuple_in_Gamma[missing_attribute]

        return result

    def imputation_via_Naive_Bayes(self, dataframe_tuples):
        print("-------imputation via Naive Bayes starts----")
        count = 0
        for index, tuple in dataframe_tuples.iterrows():
            if count % 1000 == 0:
                print("the number of tuples handled: ", count)
            if len(str(tuple['id']).split('||')) > 1:  # don't handle the tuple, where DecideTS(tuple) = false
                continue
            missing_attributes = []
            complete_attribtues = []
            for attribute in tuple.index.values.tolist():
                if attribute == 'id':
                    continue
                if not pd.isnull(tuple[attribute]):
                    complete_attribtues.append(attribute)
                else:
                    missing_attributes.append(attribute)

            shortest_distance = 10000 
            nearest_tuple_idx = 0
            current_idx = 0
            for no_use_index, tuple_in_Gamma in self.Gamma.iterrows():
                current_dist = 0
                for attribute in complete_attribtues:
                    if tuple[attribute] != tuple_in_Gamma[attribute]:
                        current_dist += 1
                if current_dist < shortest_distance:
                    shortest_distance = current_dist
                    nearest_tuple_idx = current_idx
                current_idx += 1

            for miss_att in missing_attributes:
                imputed_value = ''
                if not pd.isnull(self.Gamma.loc[nearest_tuple_idx][miss_att]):
                    imputed_value = self.Gamma.loc[nearest_tuple_idx][miss_att]
                dataframe_tuples.loc[index, miss_att] = imputed_value

            count += 1

        print("-------imputation via Naive Bayes ends----")

        return dataframe_tuples

    def imputation_via_Naive_Bayes_batch(self, dataframe_tuples, exist_conflict):
        print("------- imputation via Naive Bayes starts -------")
        split_impute_indices = np.where(exist_conflict == 1)[0]
        repair_impute_indices = np.where(exist_conflict == -1)[0]
        all_impute_indices = np.array(list(set(split_impute_indices).union(set(repair_impute_indices))))

        def impute_via_Naive_Bayes(tuple):
            # get complete attributes and missing attributes
            complete_attributes = tuple.loc[pd.notnull(tuple)].index.values.tolist()
            missing_attributes = tuple.loc[pd.isnull(tuple)].index.values.tolist()
            if "id" in complete_attributes:
                complete_attributes.remove("id")
            if "id" in missing_attributes:
                missing_attributes.remove("id")

            if len(missing_attributes) == 0 or len(complete_attributes) == 0:
                return tuple

            # find the tuple in Gamma with the nearest distance from current tuple
            distance = np.array([0] * self.Gamma.shape[0])
            for attr in complete_attributes:
                not_equal = (self.Gamma[attr] != tuple[attr]).values
                not_equal = np.array([1 if not_equal[i] == True else 0 for i in range(not_equal.shape[0])])
                distance = distance + not_equal
            min_index = distance.argmin()

            # impute with values of tuples with the min distance
            min_distance_tuple_in_Gamma = self.Gamma.iloc[min_index]
            for attr in missing_attributes:
                tuple[attr] = min_distance_tuple_in_Gamma[attr]

            return tuple

        dataframe_tuples.loc[all_impute_indices] = dataframe_tuples.loc[all_impute_indices].parallel_apply(lambda x: impute_via_Naive_Bayes(x), axis=1)

        print("------- imputation via Naive Bayes finished -------")
        return dataframe_tuples

    def compute_attribute_value_with_highest_frequency_in_Gamma(self): # invoke this function prior to imputation_via_frequency 
        for attribute in self.Gamma.columns:
            distinct_value_length = self.Gamma.groupby(attribute).size().sort_values().size # the number of distinct attributes after group-by
            self.Gamma_attribute_value_with_highest_frequency[attribute] = self.Gamma.groupby(attribute).size().sort_values().index[distinct_value_length-1] # add the value with the highest frequency in the current attribute domain

    def imputation_via_frequency(self, dataframe_tuples): # fill the missing_attribute with the value with the highest frequency of Gamma in corresponding attribute
        print("-------imputation via frequency start----")
        if len(self.Gamma_attribute_value_with_highest_frequency) < 1:
            self.compute_attribute_value_with_highest_frequency_in_Gamma()
        count = 0
        for index, tuple in dataframe_tuples.iterrows():
            if count % 1000 == 0:
                print("the number of tuples handled: ", count)
            if len(str(tuple['id']).split('||')) > 1:  # don't handle the tuple, where DecideTS(tuple) = false
                continue
            for attribute in dataframe_tuples.columns.values.tolist():
                if pd.isnull(tuple[attribute]):
                    dataframe_tuples.loc[index, attribute] = self.Gamma_attribute_value_with_highest_frequency[attribute]
            count += 1
        print("-------imputation via Naive Bayes end----")
        return dataframe_tuples

    def sort_attributes(self, missing_attributes):  # sort missing attributes based on their power for helping other missing attributes
        sorted_attributes = []
        attribute_pos = {}
        dominance_count = []
        for idx in range(len(missing_attributes)):
            attribute_pos[missing_attributes[idx]] = idx
            dominance_count.append(0)

        for attribute in missing_attributes:
            for ree in self.useful_REEs:
                attributes_in_X_of_rule = ree.get_distinct_attributes_in_X()
                if attribute in attributes_in_X_of_rule:
                    pos = attribute_pos[attribute]
                    dominance_count[pos] += 1

        for idx in range(len(dominance_count)):
            max_pos = np.array(dominance_count).argmax()
            max_count = dominance_count[max_pos]
            sorted_attributes.append(missing_attributes[max_pos])
            dominance_count[max_pos] = -1  # checked

        sorted_attributes.reverse()
        return sorted_attributes

    # a heuristic implementation of the chase for completing an incomplete tuple
    def chase(self, tuple, useREE, useKG, useMd):
        # the following codes are for n-round chase (n >=2), if conflicts exist in the last round
        # if(self.A_star is not None and len(self.SA > 0)):
        #     conflict_att_last_round = self.A_star
        #     for h in self.SA[conflict_att_last_round]:

        # ------ obtain the complete and missing attributes of the tuple ---------------------------
        missing_attributes = set()
        complete_attributes = set()
        for attribute_name in tuple.index:
            if pd.isnull(tuple[attribute_name]):
                missing_attributes.add(attribute_name)
            else:
                complete_attributes.add(attribute_name)

        # -------set the position of the sublist in SA for each attribute-----------------------
        missing_attributes = list(missing_attributes)
        # missing_attributes_copy = missing_attributes # a copy for helping updating self.SA later
        # attribute_pos_in_SA_dict = {}
        # for idx in len(missing_attributes):
        #     attribute_pos_in_SA_dict[''+missing_attributes[idx]+''] = idx

        # ------ remove REEs from Sigma that not for imputing missing attributes---------------------------
        if useREE is True:
            self.initialize_useful_REEs(tuple, missing_attributes)

        # ------ sort the missing attributes based on their power for determining other missing attributes
        if useREE is True:
            missing_attributes = self.sort_attributes(missing_attributes)

        # ------ initialize global structure; and initialize SA[pos] as [] for each missing attribute in missing_attributes
        self.re_init()
        self.SA_initialize(missing_attributes)

        # ------ data imputation withx the chase start ---------------------------
        # while len(self.useful_REEs) > 0 and len(missing_attributes) > 0:
        while len(missing_attributes) > 0:
            current_attribute = missing_attributes.pop()  # the current missing attribute to be imputed
            imputed_value = None
            if useREE is True:
                for ree in self.useful_REEs:
                    # e_attribute = ree.get_e()[0].split('.')[1]  # the dependent attribute in predicate e
                    e_attribute = ree.get_RHS().get_attr1()
                    if e_attribute != current_attribute:  # if this REE is not for imputing tuple[current_attribute], then check next REE
                        continue
                    # current_ree = self.useful_REEs.pop()  # wrong
                    attributes_in_X_of_rule = ree.get_distinct_attributes_in_X()
                    if not all(att in complete_attributes for att in attributes_in_X_of_rule):  # if complete_attributes cannot cover all attributes in preconditions X of rule, then this rule cannot be satisfied and enforced
                        continue
                        # for other_missing_attribute in missing_attributes_copy:
                        #     if (other_missing_attribute != current_attribute and other_missing_attribute in attributes_in_X_of_rule):
                        #         self.add_SA(current_ree, attribute_pos_in_SA_dict[''+other_missing_attribute+''])
                    else:  # this REE can be used for imputing tuple[current_attribute], though some preconditions X may not be satisfied yet
                        imputed_value = self.imputation_via_logic(tuple, current_attribute, ree)
                        if not pd.isnull(imputed_value):  # logic rule returns a concrete value
                            tuple[current_attribute] = imputed_value
                            break
                        # else: # logic rule fails
                        #     imputed_value = self.imputation_via_HER(tuple, current_attribute)
                        #     if (not pd.isnull(imputed_value)): # HER returns a concrete value
                        #         tuple[''+current_attribute+''] = imputed_value
                        #         break
                        #     else:
                        #         imputed_value = self.imputation_via_Md(tuple, current_attribute)
                        #         if (not pd.isnull(imputed_value)): # M_d model returns a concrete value
                        #             tuple[''+current_attribute+''] = imputed_value
                        #             break

                        # complete_attributes.add(current_attribute) # update the complete attribute

                        # if (pd.isnull(imputed_value)): # all imputation approaches fail
                        #     tuple[''+current_attribute+''] = None
                        #     break

                        # for activated_ree in list(self.SA[attribute_pos_in_SA_dict[''+current_attribute+'']]): # check the activated rules by enforcing the current_ree
                        #     attributes_in_X_of_rule =  activated_ree.get_distinct_attributes_in_X()
                        #     if (not all(att in complete_attributes for att in attributes_in_X_of_rule)):
                        #         continue
                        #     else:
                        #         self.SA[attribute_pos_in_SA_dict[''+current_attribute+'']].remove(activated_ree) # this rule is ready for enforcing
                        #         self.useful_REEs.append(activated_ree)
                        #         pass

                        # for removed_ree in list(self.useful_REEs): # remove the REEs for imputing tuple[current_attribute], since it is already imputed
                        #     e_attribute = removed_ree.get_e()[0].split('.')[1] # the dependent attribute in predicate e
                        #     if (e_attribute == current_attribute): # if this REE is not for imputing tuple[current_attribute], then check next REE
                        #         self.useful_REEs.remove(removed_ree)

            if pd.isnull(imputed_value) and useKG is True:  # logic imptuation fails
                imputed_value = self.imputation_via_HER(tuple, complete_attributes, current_attribute)
                # print ('--------------------------------test_1-------------------------------------')
                # print(imputed_value)
                # print(current_attribute)
                if not pd.isnull(imputed_value):  # HER returns a concrete value
                    tuple[current_attribute] = imputed_value

            if pd.isnull(imputed_value) and useMd is True:  # HER fails
                imputed_value = self.imputation_via_Md(tuple, current_attribute)[0]
                # print ('--------------------------------test_2-------------------------------------')
                # print(imputed_value)
                # print(current_attribute)
                if not pd.isnull(imputed_value):  # M_d model returns a concrete value
                    tuple[current_attribute] = imputed_value

            complete_attributes.add(current_attribute)  # update the complete attribute

            if pd.isnull(imputed_value):  # all imputation approaches fail
                tuple[current_attribute] = None

            # remove the REEs contribute to current_attribute
            if useREE is True:
                rm_rees = []
                for ree in self.useful_REEs:  # remove the REEs for imputing tuple[current_attribute], since it is already imputed
                    # e_attribute = removed_ree.get_e()[0].split('.')[1]  # the dependent attribute in predicate e
                    e_attribute = ree.get_RHS().get_attr1()
                    if e_attribute == current_attribute:  # if this REE is not for imputing tuple[current_attribute], then check next REE
                        rm_rees.append(ree)
                for ree in rm_rees:
                    self.useful_REEs.remove(ree)

        self.update_Gamma(tuple)

        self.tuples_res.append(tuple)

    # input the splitting tuples
    def chase_batch(self, tuples, useREE, useKG, useMd, main_attributes, exist_conflict, output_tmp_files_Md_dir, her, Md_model, max_chase_round, if_update_Gamma, impute_all_empty_values, tuple_status):
        split_impute_indices = np.where(exist_conflict == 1)[0]
        repair_impute_indices = np.where(exist_conflict == -1)[0]
        all_impute_indices = np.array(list(set(split_impute_indices).union(set(repair_impute_indices))))

        # obtain the columns that might be imputed
        all_attrs = tuples.columns.values
        # main_attrs = main_attributes.split("||")
        # reserved_attrs = [attr for attr in all_attrs if attr not in main_attrs and attr != "id"]
        reserved_attrs = [attr for attr in all_attrs if attr != "id"]

        round = 0
        active = True
        last_updated = pd.Series([True] * all_impute_indices.shape[0])  # whether the tuples be updated in the last round; In round 0, it should be all True
        current_updated = pd.Series([False] * all_impute_indices.shape[0])  # whether the tuples be updated in the current round; In round 0, it should be all False
        last_updated.index = all_impute_indices
        current_updated.index = all_impute_indices
        while active is True:
            active = False
            if round >= max_chase_round:
                break
            print("chase round: ", round)

            # the indices of tuples that has been updated in the last round
            last_updated_indices = last_updated.loc[last_updated == True].index.values
            # reset the current_updated to be all False
            current_updated.loc[:] = False

            for current_attr in reserved_attrs:
                # print("-----------------------------------------------------------------")
                # print("current_attr:", current_attr)
                tuples_to_check = tuples.loc[last_updated_indices]
                reserved_indices = tuples_to_check.loc[tuples_to_check[current_attr].isnull()].index.values
                if reserved_indices.shape[0] == 0:
                    continue
                flag_all_filled = False

                # -------------------------------- useREE --------------------------------
                if useREE is True:
                    # start_time = timeit.default_timer()
                    # print("=== REE size:", reserved_indices.shape[0], "===")
                    useful_single_rees, useful_bi_rees = filter_useful_rees(self.Sigma, current_attr)
                    # 1. use single-variable REEs
                    for ree in useful_single_rees:
                        tuples_to_check = tuples.loc[last_updated_indices]
                        reserved_indices = tuples_to_check.loc[tuples_to_check[current_attr].isnull()].index.values
                        if reserved_indices.shape[0] == 0:
                            flag_all_filled = True
                            break
                        for predicate in ree.get_currents():
                            attr = predicate.get_attr1()
                            constant = predicate.get_constant()
                            reserved_tuples_check = tuples.loc[reserved_indices]
                            reserved_indices = reserved_tuples_check.loc[reserved_tuples_check[attr].astype(str) == constant].index.values
                            if reserved_indices.shape[0] == 0:
                                break
                        if reserved_indices.shape[0] != 0:  # the indices of tuples that satisfy X of ree
                            rhs = ree.get_RHS()
                            value = rhs.get_constant()
                            if pd.notnull(value):
                                tuples.loc[reserved_indices, current_attr] = value
                                active = True
                                current_updated.loc[reserved_indices] = True
                                # update tuple_status
                                tuple_status_check = tuple_status.loc[reserved_indices, current_attr]
                                repaired_indices = tuple_status_check.loc[tuple_status_check == 2].index.values
                                origin_null_indices = np.array(list(set(reserved_indices).difference(set(repaired_indices))))
                                tuple_status.loc[repaired_indices, current_attr] = 4  # repair & impute
                                tuple_status.loc[origin_null_indices, current_attr] = 3  # impute
                                # print("\n...... MI-1, attribute [ " + current_attr + " ] was assigned by value [ " + str(value) + " ], by REE: " + ree.print_rule())  # track

                    # 2. use bi-variable REEs
                    for ree in useful_bi_rees:
                        if flag_all_filled is True:
                            break
                        rhs = ree.get_RHS()
                        if rhs.get_type() == "constant" and rhs.get_index1() == 1:
                            continue
                        tuples_to_check = tuples.loc[last_updated_indices]
                        reserved_indices = tuples_to_check.loc[tuples_to_check[current_attr].isnull()].index.values
                        if reserved_indices.shape[0] == 0:
                            flag_all_filled = True
                            break
                        value2tid_tuples, value2tid_GT = {}, {}
                        constants_in_X = [[], []]
                        # get key
                        key_attributes_non_constant = []
                        for predicate in ree.get_currents():
                            if predicate.get_type() == "non-constant":
                                key_attributes_non_constant.append(predicate.get_attr1())
                            else:
                                pre_tid = predicate.get_index1()
                                constants_in_X[pre_tid].append((predicate.get_attr1(), predicate.get_constant()))  # (A, a)
                        # get value of tuples
                        # (1) first filter tuples that not satisfy the constant predicates
                        for attr, v in constants_in_X[0]:  # constant predicates
                            tuples_check_constants = tuples.loc[reserved_indices]
                            reserved_indices = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
                            if reserved_indices.shape[0] == 0:
                                break
                        if reserved_indices.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                            continue
                        # (2) then construct dict by non-constant predicates
                        for value, df in tuples.loc[reserved_indices].groupby(key_attributes_non_constant):
                            value2tid_tuples[value] = df
                        if len(value2tid_tuples) == 0:
                            continue
                        # get value of GT
                        # (1) first filter tuples that not satisfy the constant predicates
                        reserved_indices_GT = self.Gamma.index
                        for attr, v in constants_in_X[1]:  # constant predicates
                            tuples_check_constants = self.Gamma.loc[reserved_indices_GT]
                            reserved_indices_GT = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
                            if reserved_indices_GT.shape[0] == 0:
                                break
                        if reserved_indices_GT.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                            continue
                        # (2) then construct dict by non-constant predicates
                        for value, df in self.Gamma.loc[reserved_indices_GT].groupby(key_attributes_non_constant):
                            value2tid_GT[value] = df  # df.index.values
                        if len(value2tid_GT) == 0:
                            continue
                        # apply constant Y
                        if rhs.get_type() == "constant":
                            for key in value2tid_tuples.keys():
                                if key not in value2tid_GT.keys():
                                    continue
                                indices = value2tid_tuples[key].index.values
                                if pd.notnull(rhs.get_constant()):
                                    tuples.loc[indices, current_attr] = rhs.get_constant()
                                    active = True
                                    current_updated.loc[indices] = True
                                    # update tuple_status
                                    tuple_status_check = tuple_status.loc[indices, current_attr]
                                    repaired_indices = tuple_status_check.loc[tuple_status_check == 2].index.values
                                    origin_null_indices = np.array(list(set(indices).difference(set(repaired_indices))))
                                    tuple_status.loc[repaired_indices, current_attr] = 4  # repair & impute
                                    tuple_status.loc[origin_null_indices, current_attr] = 3  # impute
                                    # print("\n...... MI-1, attribute [ " + current_attr + " ] was assigned by value [ " + rhs.get_constant() + " ], by REE: " + ree.print_rule())  # track
                        # apply non-constant Y
                        else:
                            for key in value2tid_tuples.keys():
                                if key not in value2tid_GT.keys():
                                    continue
                                indices = value2tid_tuples[key].index.values
                                series_GT = value2tid_GT[key][current_attr]
                                valid_idx = series_GT.first_valid_index()
                                value = series_GT.loc[valid_idx] if valid_idx is not None else None  # choose the first non-null value for imputation!  can be changed to obtain all values for imputation
                                if pd.notnull(value):
                                    tuples.loc[indices, current_attr] = value
                                    active = True
                                    current_updated.loc[indices] = True
                                    # update tuple_status
                                    tuple_status_check = tuple_status.loc[indices, current_attr]
                                    repaired_indices = tuple_status_check.loc[tuple_status_check == 2].index.values
                                    origin_null_indices = np.array(list(set(indices).difference(set(repaired_indices))))
                                    tuple_status.loc[repaired_indices, current_attr] = 4  # repair & impute
                                    tuple_status.loc[origin_null_indices, current_attr] = 3  # impute
                                    # print("\n...... MI-1, attribute [ " + current_attr + " ] was assigned by value [ " + str(value) + " ], by REE: " + ree.print_rule())  # track
                    # print("1. REE time:", timeit.default_timer() - start_time)

                tuples_to_check = tuples.loc[last_updated_indices]
                reserved_indices = tuples_to_check.loc[tuples_to_check[current_attr].isnull()].index.values
                if reserved_indices.shape[0] == 0:
                    flag_all_filled = True

                # -------------------------------- useKG --------------------------------
                if flag_all_filled is False and useKG is True:
                    # if self.varyKG is True and self.ratio < 1.0:
                    #     impute_size = int(reserved_indices.shape[0] * self.ratio)
                    #     ramdom_idx = random.sample(range(0, reserved_indices.shape[0]), impute_size)
                    #     reserved_indices = reserved_indices[ramdom_idx]
                    # start_time = timeit.default_timer()
                    # print("=== HER size:", reserved_indices.shape[0], "===")
                    success, imputed_results, updated_indices = her.imputation_via_HER_batch(tuples.loc[reserved_indices], current_attr, impute_all_empty_values)
                    # print("2. HER time:", timeit.default_timer() - start_time)
                    if success is True:
                        if impute_all_empty_values is False:
                            tuples.loc[reserved_indices, current_attr] = imputed_results
                            # print("\n...... MI-2, use HER to impute one attribute: the attribute [ " + current_attr + " ] was assigned by value [ " + str(imputed_results) + " ]")  # track
                        else:
                            # print("\n...... MI-2, use HER to impute all attributes. Before assignment, tuple:\n", tuples)  # track
                            tuples.loc[reserved_indices] = imputed_results
                            # print("\n...... MI-2, After assignment by HER, tuple: \n", tuples)  # track
                        active = True
                        current_updated.loc[updated_indices] = True
                        # update tuple_status
                        tuple_status_check = tuple_status.loc[updated_indices, current_attr]
                        repaired_indices = tuple_status_check.loc[tuple_status_check == 2].index.values
                        origin_null_indices = np.array(list(set(updated_indices).difference(set(repaired_indices))))
                        tuple_status.loc[repaired_indices, current_attr] = 4  # repair & impute
                        tuple_status.loc[origin_null_indices, current_attr] = 3  # impute

                tuples_to_check = tuples.loc[last_updated_indices]
                reserved_indices = tuples_to_check.loc[tuples_to_check[current_attr].isnull()].index.values
                if reserved_indices.shape[0] == 0:
                    flag_all_filled = True

                # -------------------------------- useMd --------------------------------
                '''
                if flag_all_filled is False and useMd is True:
                    # start_time = timeit.default_timer()
                    # print("=== Md size:", reserved_indices.shape[0], "===")
                    imputed_values = Md_model.impute_Md_batch(tuples.loc[reserved_indices], current_attr)
                    # print("3. Md time:", timeit.default_timer() - start_time)
                    imputed_values = np.array(imputed_values)
                    impute_not_null_indices = np.where(imputed_values != "")[0]
                    impute_not_null_indices_in_tuples = np.array([reserved_indices[i] for i in impute_not_null_indices])
                    if impute_not_null_indices_in_tuples.shape[0] != 0:
                        tuples.loc[impute_not_null_indices_in_tuples, current_attr] = imputed_values[impute_not_null_indices]
                        active = True
                        current_updated.loc[impute_not_null_indices_in_tuples] = True
                    # tmp_output_path = output_tmp_files_Md_dir + "tuples_splitting_" + current_attr + ".csv"
                    # tuples.to_csv(tmp_output_path, index=False)
                    # tmp_input_path = output_tmp_files_Md_dir + "tuples_splitting_" + current_attr + "_Md.csv"
                    # exec_str = "python CorrelationModel/wiki_m_d.py -input_splitting_tuples_path " + tmp_output_path + " -output_splitting_tuples_path " + tmp_input_path
                    # os.system(exec_str)
                    # updated_column_Md = pd.read_csv(tmp_input_path)[current_attr]
                    # if (tuples[current_attr] != updated_column_Md).sum() > 0:
                    #     tuples[current_attr] = updated_column_Md
                    #     active = True
                '''

                # print("current_attr: ", current_attr, ", active: ", active)

            # ------------------------------ update Gamma ------------------------------
            if if_update_Gamma is True:
                tuples_to_check = tuples.loc[last_updated_indices]
                contains_null_indices = tuples_to_check[tuples_to_check.isnull().T.any()].index.values
                # valid_indices = [i for i in last_updated_indices if i not in contains_null_indices]
                valid_indices = list(set(last_updated_indices).difference(set(contains_null_indices)))
                self.update_Gamma_by_tuples(tuples.loc[valid_indices])

            round = round + 1
            last_updated = current_updated

        if useMd is True:
            updated = []
            for current_attr in reserved_attrs:
                tuples_to_check = tuples.loc[all_impute_indices]
                reserved_indices = tuples_to_check.loc[tuples_to_check[current_attr].isnull()].index.values
                if reserved_indices.shape[0] == 0:
                    continue
                imputed_values = Md_model.impute_Md_batch(tuples_to_check.loc[reserved_indices], current_attr)
                imputed_values = np.array(imputed_values)
                impute_not_null_indices = np.where(imputed_values != "")[0]
                impute_not_null_indices_in_tuples = np.array([reserved_indices[i] for i in impute_not_null_indices])
                if impute_not_null_indices_in_tuples.shape[0] != 0:
                    tuples.loc[impute_not_null_indices_in_tuples, current_attr] = imputed_values[impute_not_null_indices]
                    updated = updated + [i for i in impute_not_null_indices_in_tuples]
                    # update tuple_status
                    tuple_status_check = tuple_status.loc[impute_not_null_indices_in_tuples, current_attr]
                    repaired_indices = tuple_status_check.loc[tuple_status_check == 2].index.values
                    origin_null_indices = np.array(list(set(impute_not_null_indices_in_tuples).difference(set(repaired_indices))))
                    tuple_status.loc[repaired_indices, current_attr] = 4  # repair & impute
                    tuple_status.loc[origin_null_indices, current_attr] = 3  # impute
                    # print("\n...... MI-2, use Md to impute: the attribute [ " + current_attr + " ] was assigned by value [ " + str(imputed_values[impute_not_null_indices][0]) + " ]")  # track

            if if_update_Gamma is True:
                tuples_to_check = tuples.loc[np.array(set(updated))]
                contains_null_indices = tuples_to_check[tuples_to_check.isnull().T.any()].index.values
                valid_indices = list(set(updated).difference(set(contains_null_indices)))
                self.update_Gamma_by_tuples(tuples.loc[valid_indices])

        return tuples, tuple_status

    def obtain_Gamma_size(self):
        return self.Gamma.drop_duplicates().reset_index(drop=True).shape[0]

    # after imputation, write tuples into file
    def write_to_file(self, path):
        self.tuples_res = pd.DataFrame(self.tuples_res)  # tuples_res: this result only contains the tuples that are split and imputed
        self.tuples_res.to_csv(path, index=False)

    def write_groudtruth_to_file(self, path):
        self.Gamma.to_csv(path)

    # check all valuations; re-chase from the beginning when a conflict appears
    def chase_all_valuations(self, tuple, useREE, useKG, useMd, max_chase_round, her, Md_model):
        # ------ obtain the complete and missing attributes of the tuple ---------------------------
        index_tuple = tuple.index[0]
        missing_attributes = set()
        complete_attributes = set()
        for attribute_name in tuple.columns:
            if pd.isnull(tuple.loc[index_tuple, attribute_name]):
                missing_attributes.add(attribute_name)
            else:
                complete_attributes.add(attribute_name)

        missing_attributes = list(missing_attributes)

        pos = [idx for idx in range(len(missing_attributes))]

        attribute_pos = pd.DataFrame(index=missing_attributes, data=pos) # record the pos of missing attributes in missing_attributes

        confidence = [0. for idx in range(len(missing_attributes))] # the confidence of current imputed value

        imputed_values = [[] for idx in range(len(missing_attributes))] # the imputed attribute values

        # ------ remove REEs from Sigma that not for imputing missing attributes---------------------------
        if useREE is True:
            # no_filter_via_missing = []
            # self.initialize_useful_REEs(tuple, no_filter_via_missing) # do not filter rules based on missing attributes
            self.initialize_useful_REEs(tuple, missing_attributes) # filter rules based on missing attributes

        # ------ sort the missing attributes based on their power for determining other missing attributes
        # if useREE is True:
        #     missing_attributes = self.sort_attributes(missing_attributes)

        # ------ data imputation withx the chase start ---------------------------
        conflict_flag = False
        no_more_round = False
        round = 0
        while True:
            # if round >= max_chase_round:
            if round >= max_chase_round - 1:
                no_more_round = True
            print("chase round: ", round)
            for current_attribute in missing_attributes:
                imputed_value = None
                # tuple_to_apply = tuple
                tuple_to_apply = copy.deepcopy(tuple)
                if useREE is True:
                    for ree in self.useful_REEs:
                        # e_attribute = ree.get_e()[0].split('.')[1]  # the dependent attribute in predicate e
                        e_attribute = ree.get_RHS().get_attr1()
                        if e_attribute != current_attribute:  # if this REE is not for imputing tuple[current_attribute], then check next REE
                            continue
                        # current_ree = self.useful_REEs.pop()  # wrong
                        attributes_in_X_of_rule = ree.get_distinct_attributes_in_X()
                        if not all(att in complete_attributes for att in attributes_in_X_of_rule):  # if complete_attributes cannot cover all attributes in preconditions X of rule, then this rule cannot be satisfied and enforced
                            continue
                        else:  # this REE can be used for imputing tuple[current_attribute], though some preconditions X may not be satisfied yet
                            # imputed_value = self.imputation_via_logic(tuple, current_attribute, ree)
                            imputed_value = self.imputation_via_logic(tuple_to_apply, current_attribute, ree)
                            if not pd.isnull(imputed_value):  # logic rule returns a concrete value
                                # current_confidence = 1.0
                                if pd.isnull(tuple.loc[index_tuple, current_attribute]): # the first time to fill a missing attribute
                                    tuple[current_attribute] = imputed_value
                                    # confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                                    imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value)
                                # elif imputed_value in imputed_values[attribute_pos.loc[current_attribute][0]] and confidence[attribute_pos.loc[current_attribute][0]] != current_confidence: # a conflict case. imputed before, but with different confidences;
                                #     if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence: # if the new confidence is higher, then update the confidence
                                #         confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                                #     conflict_flag = True
                                #     if no_more_round is False:
                                #         break # conflict
                                elif imputed_value not in imputed_values[attribute_pos.loc[current_attribute][0]]: # a conflict case. a new imputed value, different from previous imputed value
                                    # if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence: # update the value and confidence if the newly imputed value has higher confidence
                                    #     tuple[current_attribute] = imputed_value
                                    #     confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                                    imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value) 
                                    conflict_flag = True
                                    if no_more_round is False:
                                        break # conflict
                    if conflict_flag is True and no_more_round is False:
                        break # conflict

                if useKG is True:  # apply HER
                    # imputed_value = self.imputation_via_HER(tuple, complete_attributes, current_attribute)
                    # success, imputed_value = her.imputation_via_HER_batch(tuple, current_attribute)
                    success, imputed_value, no_use = her.imputation_via_HER_batch(tuple_to_apply, current_attribute, impute_all_empty_values=False)
                    # imputed_value = imputed_value[0]
                    if imputed_value is not None:
                        # print("-----------KG debug-----------")
                        # print(imputed_value)
                        # print("imputed_value.index", imputed_value.index)
                        # print("imputed_value.index[0]", imputed_value.index[0])
                        index_key = imputed_value.index[0]
                        imputed_value = imputed_value[index_key]
                    if success is True and imputed_value is not None:
                        # print ('--------------------------------test_1-------------------------------------')
                        if not pd.isnull(imputed_value):  # HER returns a concrete value
                            # current_confidence = 1.0  # assume the KG is trustable
                            # print("-----------KG debug-----------")
                            # print("len(tuple[current_attribute])", len(tuple[current_attribute]))
                            # print("tuple[current_attribute]", tuple[current_attribute])
                            # print("tuple[current_attribute].index[0]", tuple[current_attribute].index[0])
                            # print("tuple[current_attribute][tuple[current_attribute].index[0]]", tuple[current_attribute][tuple[current_attribute].index[0]])
                            # if pd.isnull(tuple[current_attribute]): # the first time to fill a missing attribute
                            #     tuple[current_attribute] = imputed_value
                            if pd.isnull(tuple[current_attribute][tuple[current_attribute].index[0]]): # the first time to fill a missing attribute
                                tuple[current_attribute][tuple[current_attribute].index[0]] = imputed_value
                                # confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                                imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value)
                            # elif imputed_value in imputed_values[attribute_pos.loc[current_attribute][0]] and confidence[attribute_pos.loc[current_attribute][0]] != current_confidence: # a conflict case. imputed before, but with different confidences;
                            #     if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence: # if the new confidence is higher, then update the confidence
                            #         confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                            #     conflict_flag = True
                            #     if no_more_round is False:
                            #         break # conflict
                            elif imputed_value not in imputed_values[attribute_pos.loc[current_attribute][0]]: # a conflict case. a new imputed value, different from previous imputed value
                                # if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence: # update the value and confidence if the newly imputed value has higher confidence
                                #     # tuple[current_attribute] = imputed_value
                                #     tuple[current_attribute][tuple[current_attribute].index[0]] = imputed_value
                                #     confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                                imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value)
                                conflict_flag = True
                                if no_more_round is False:
                                    break # conflict

                '''
                if useMd is True:  # apply Md
                    # imputed_value_and_confidence = self.imputation_via_Md(tuple, current_attribute)
                    # imputed_value = imputed_value_and_confidence[0]
                    # imputed_value = Md_model.impute_Md_batch(tuple, current_attribute)[0]
                    imputed_value = Md_model.impute_Md_batch(tuple_to_apply, current_attribute)[0]
                    if not pd.isnull(imputed_value):  # M_d model returns a concrete value
                        # current_confidence = imputed_value_and_confidence[1] # the confidence of the imputed value returned by Md
                        current_confidence = 0.8  # the confidence of the imputed value returned by Md
                        if pd.isnull(tuple.loc[index_tuple, current_attribute]): # the first time to fill a missing attribute
                            tuple[current_attribute] = imputed_value
                            # print("-----------debug 2 ---------------")
                            # print("current_attribute -->", current_attribute)
                            # print("attribute_pos.loc[current_attribute][0] -->", attribute_pos.loc[current_attribute][0])
                            # print("confidence[attribute_pos.loc[current_attribute][0]]  -->", confidence[attribute_pos.loc[current_attribute][0]])
                            confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                            imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value)
                        elif imputed_value in imputed_values[attribute_pos.loc[current_attribute][0]] and confidence[attribute_pos.loc[current_attribute][0]] != current_confidence: # a conflict case. imputed before, but with different confidences;
                            if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence: # if the new confidence is higher, then update the confidence
                                confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                            conflict_flag = True
                            if no_more_round is False:
                                break # conflict
                        elif imputed_value not in imputed_values[attribute_pos.loc[current_attribute][0]]: # a conflict case. a new imputed value, different from previous imputed value
                            if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence: # update the value and confidence if the newly imputed value has higher confidence
                                tuple[current_attribute] = imputed_value
                                confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                            imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value) 
                            conflict_flag = True
                            if no_more_round is False:
                                break # conflict
                '''

                # complete_attributes.add(current_attribute)  # update the complete attribute

                # if pd.isnull(imputed_value):  # all imputation approaches fail
                #     tuple[current_attribute] = None
                # else:
                #     complete_attributes.add(current_attribute)  # update the complete attribute
                if pd.notnull(imputed_value):  # all imputation approaches fail
                    complete_attributes.add(current_attribute)  # update the complete attribute

                # remove the REEs contribute to current_attribute
                # if useREE is True:
                #     rm_rees = []
                #     for ree in self.useful_REEs:  # remove the REEs for imputing tuple[current_attribute], since it is already imputed
                #         # e_attribute = removed_ree.get_e()[0].split('.')[1]  # the dependent attribute in predicate e
                #         e_attribute = ree.get_RHS().get_attr1()
                #         if e_attribute == current_attribute:  # if this REE is not for imputing tuple[current_attribute], then check next REE
                #             rm_rees.append(ree)
                #     for ree in rm_rees:
                #         self.useful_REEs.remove(ree)

            if no_more_round:
                break

            round = round + 1
            if conflict_flag:
                conflict_flag = False
                continue # another round of chase
            # else:
            #     break # all missing attributes are filled without conflicts

            # break # for speed purpose

        if useMd is True:  # apply Md
            for current_attribute in missing_attributes:
                tuple_to_apply = copy.deepcopy(tuple)
                imputed_value = Md_model.impute_Md_batch(tuple_to_apply, current_attribute)[0]
                if not pd.isnull(imputed_value):  # M_d model returns a concrete value
                    # current_confidence = 0.8  # the confidence of the imputed value returned by Md
                    if pd.isnull(tuple.loc[index_tuple, current_attribute]):  # the first time to fill a missing attribute
                        tuple[current_attribute] = imputed_value
                        # confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                        imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value)
                    # elif imputed_value in imputed_values[attribute_pos.loc[current_attribute][0]] and confidence[attribute_pos.loc[current_attribute][0]] != current_confidence:  # a conflict case. imputed before, but with different confidences;
                    #     if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence:  # if the new confidence is higher, then update the confidence
                    #         confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                    elif imputed_value not in imputed_values[attribute_pos.loc[current_attribute][0]]:  # a conflict case. a new imputed value, different from previous imputed value
                        # if confidence[attribute_pos.loc[current_attribute][0]] < current_confidence:  # update the value and confidence if the newly imputed value has higher confidence
                        #     tuple[current_attribute] = imputed_value
                        #     confidence[attribute_pos.loc[current_attribute][0]] = current_confidence
                        imputed_values[attribute_pos.loc[current_attribute][0]].append(imputed_value)

        # self.update_Gamma(tuple)

        # self.tuples_res.append(tuple)
