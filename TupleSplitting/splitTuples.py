from TupleSplitting.rule import REELogic, REEMc
from TupleSplitting.func import *
import pandas as pd
import numpy as np
import copy
import timeit

pd.set_option('display.max_columns', None)
np.random.seed(7654321)

predictor, wikidata_embed, wikidata_dic, item_dict, item_dict_reverse = None, None, None, None, None


def load_REEs(ree_path):
    f = open(ree_path, "r", encoding='utf-8')
    lines = f.readlines()
    f.close()
    rees = []
    for line in lines:
        ree = REELogic()
        ree.load_X_and_e(line)
        rees.append(ree)
    return rees


def load_Mc(Mc_path):
    f = open(Mc_path, "r", encoding='utf-8')
    lines = f.readlines()
    f.close()
    rees_Mc = []
    for line in lines:
        M_c = REEMc()
        M_c.load_X(line)
        rees_Mc.append(M_c)
    return rees_Mc


def generateGroundTruth(data):
    GT = {}
    attrs = data.columns.values
    for index, row in data.iterrows():
        GT[index] = {}
        for attr in attrs:
            GT[index][attr] = row[attr]
    return GT


# split one tuple to multi tuples
def decideTS(GT, rees, Mc, tuples):
    conflicting_attributes = []
    exist_conflict = []
    for t_ in tuples:
        splitting_attrs = []
        # 1. check violations of single-variable REE rules
        for ree in rees:
            if ree.get_tuple_variable_cnt() > 1:
                continue
            violate, violate_by_attr = violateSingleRule(ree, t_)
            if violate is True:
                splitting_attrs.append(violate_by_attr)

        # 2. check violations of bi-variable REE rules
        for index, row in GT.iterrows():
            for ree in rees:
                if ree.get_tuple_variable_cnt() == 1:
                    continue
                violate, violate_by_attr = violateBiVariableREE(ree, t_, row)
                if violate is True:
                    splitting_attrs.append(violate_by_attr)

        # 3. check violations of Mc rules
        for Mc_rule in Mc:
            violate, violate_by_attr = violateSingleRule(Mc_rule, t_)
            if violate is True:
                splitting_attrs.append(violate_by_attr)

        conflicting_attributes.append(splitting_attrs)
        if len(splitting_attrs) > 0:
            exist_conflict.append(True)
        else:
            exist_conflict.append(False)

    return exist_conflict, conflicting_attributes


# split one tuple to two tuples
def decideTS_simplified(GT, rees, Mc, tuples, useREE, useMc):
    conflicting_attributes = []
    exist_conflict = []

    rees_single_variable = []
    rees_bi_variable = []
    for ree in rees:
        if ree.get_tuple_variable_cnt() == 1:
            rees_single_variable.append(ree)
        else:
            rees_bi_variable.append(ree)

    for index1, t_ in tuples.iterrows():
        violate = False
        violate_by_attr = None
        if useREE is True:
            # 1. check violations of single-variable REE rules
            for ree in rees_single_variable:
                violate, violate_by_attr = violateSingleRule(ree, t_)
                if violate is True:
                    break
            if violate is True:
                conflicting_attributes.append(violate_by_attr)
                exist_conflict.append(True)
                continue

            # 2. check violations of bi-variable REE rules
            for index2, row in GT.iterrows():
                for ree in rees_bi_variable:
                    violate, violate_by_attr = violateBiVariableREE(ree, t_, row)
                    if violate is True:
                        break
                if violate is True:
                    break
            if violate is True:
                conflicting_attributes.append(violate_by_attr)
                exist_conflict.append(True)
                continue

        if useMc is True:
            # 3. check violations of Mc rules
            for Mc_rule in Mc:
                violate, violate_by_attr = violateSingleRule(Mc_rule, t_)
                if violate is True:
                    break

        conflicting_attributes.append(violate_by_attr)
        if violate is True:
            exist_conflict.append(True)
        else:
            exist_conflict.append(False)

    return exist_conflict, conflicting_attributes


# split one tuple to two tuples - optimized
def decideTS_simplified_new(GT, rees, Mc, tuples, useREE, useMc):
    conflicting_attributes = [None] * tuples.shape[0]
    exist_conflict = [False] * tuples.shape[0]

    if useREE is True:
        rees_single_variable, rees_bi_variable = [], []
        # satisfy_cpredicates_bi_REEs = []
        for ree in rees:
            if ree.get_tuple_variable_cnt() == 1:
                rees_single_variable.append(ree)
            else:
                rees_bi_variable.append(ree)
        # for ree in rees_bi_variable:
        #     satisfy_cpredicates_bi_REEs.append(satisfyConstantPredicatesInX_forTuples(ree, tuples))

        # 1. check violations of single-variable REE rules
        for index, t_ in tuples.iterrows():
            for ree in rees_single_variable:
                violate, violate_by_attr = violateSingleRule(ree, t_)
                if violate is True:
                    conflicting_attributes[index] = violate_by_attr
                    exist_conflict[index] = True
                    break
        num_split_single = np.sum(np.array(exist_conflict) == True)
        print("after using single-variable REEs, num to be split: ", num_split_single)

        # 2. check violations of bi-variable REE rules
        for index_t_, t_ in tuples.iterrows():
            if exist_conflict[index_t_] is True:
                continue
            for idx_bi_ree in range(len(rees_bi_variable)):
                # regarded_as_t0 = satisfy_cpredicates_bi_REEs[idx_bi_ree][0]
                # regarded_as_t1 = satisfy_cpredicates_bi_REEs[idx_bi_ree][1]
                # if regarded_as_t0 is False and regarded_as_t1 is False:  # this ree can not be applied to check t_ and any other tuple in GT
                #     continue
                ree = rees_bi_variable[idx_bi_ree]
                violate = False
                for index_row, row in GT.iterrows():
                    # if regarded_as_t0 is True:
                        violate, violate_by_attr = violateBiVariableREE(ree, t_, row)
                        if violate is True:
                            conflicting_attributes[index_t_] = violate_by_attr
                            exist_conflict[index_t_] = True
                            break
                    # if regarded_as_t1 is True:
                        violate, violate_by_attr = violateBiVariableREE(ree, row, t_)
                        if violate is True:
                            conflicting_attributes[index_t_] = violate_by_attr
                            exist_conflict[index_t_] = True
                            break
                if violate is True:
                    break
        num_split_bi = np.sum(np.array(exist_conflict) == True)
        print("after using bi-variable REEs, num to be split: ", num_split_bi)

    if useMc is True:
        # 3. check violations of Mc rules
        for index, t_ in tuples.iterrows():
            if exist_conflict[index] is True:
                continue
            for Mc_rule in Mc:
                violate, violate_by_attr = violateSingleRule(Mc_rule, t_)
                if violate is True:
                    conflicting_attributes[index] = violate_by_attr
                    exist_conflict[index] = True
                    break
        num_split_Mc = np.sum(np.array(exist_conflict) == True)
        print("after using Mc rules, num to be split: ", num_split_Mc)

    return exist_conflict, conflicting_attributes


# split one tuple to two tuples - batch - GT
def decideTS_simplified_batch(GT, rees, Mc, tuples, useREE, useMc):
    conflicting_attributes = [None] * tuples.shape[0]
    exist_conflict = [False] * tuples.shape[0]

    if useREE is True:
        rees_single_variable, rees_bi_variable = [], []
        for ree in rees:
            if ree.get_tuple_variable_cnt() == 1:
                rees_single_variable.append(ree)
            else:
                rees_bi_variable.append(ree)

        # 1. check violations of single-variable REE rules
        for index_t_, t_ in tuples.iterrows():
            for ree in rees_single_variable:
                violate, violate_by_attr = violateSingleRule(ree, t_)
                if violate is True:
                    exist_conflict[index_t_] = True
                    conflicting_attributes[index_t_] = violate_by_attr
                    break
        num_split_single = np.sum(np.array(exist_conflict) == True)
        print("after using single-variable REEs, num to be split: ", num_split_single)

        # 2. check violations of bi-variable REE rules
        for index_t_, t_ in tuples.iterrows():
            if exist_conflict[index_t_] is True:
                continue
            for idx_bi_ree in range(len(rees_bi_variable)):
                ree = rees_bi_variable[idx_bi_ree]
                violate, violate_by_attr = violateBiVariableREE_ForTuples(ree, t_, GT)
                if violate is True:
                    exist_conflict[index_t_] = True
                    conflicting_attributes[index_t_] = violate_by_attr
                    break
        num_split_bi = np.sum(np.array(exist_conflict) == True)
        print("after using bi-variable REEs, num to be split: ", num_split_bi)

    if useMc is True:
        # 3. check violations of Mc rules
        for index_t_, t_ in tuples.iterrows():
            if exist_conflict[index_t_] is True:
                continue
            for Mc_rule in Mc:
                violate, violate_by_attr = violateSingleRule(Mc_rule, t_)
                if violate is True:
                    exist_conflict[index_t_] = True
                    conflicting_attributes[index_t_] = violate_by_attr
                    break
        num_split_Mc = np.sum(np.array(exist_conflict) == True)
        print("after using Mc rules, num to be split: ", num_split_Mc)

    return exist_conflict, conflicting_attributes


# split one tuple to two tuples - batch - GT and Tuples
def decideTS_simplified_batch_all(GT, rees, Mc, tuples, useREE, useMc):
    conflicting_attributes = [None] * tuples.shape[0]
    exist_conflict = [False] * tuples.shape[0]

    if useREE is True:
        rees_single_variable, rees_bi_variable = [], []
        for ree in rees:
            if ree.get_tuple_variable_cnt() == 1:
                rees_single_variable.append(ree)
            else:
                rees_bi_variable.append(ree)

        # 1. check violations of single-variable REE rules
        for ree in rees_single_variable:
            violate, violate_by_attr, violate_rule_indices = violateSingleRule_ForTuples(ree, tuples)
            if violate is True:
                for index_t_ in violate_rule_indices:
                    exist_conflict[index_t_] = True
                    conflicting_attributes[index_t_] = violate_by_attr

        num_split_single = np.sum(np.array(exist_conflict) == True)
        print("after using single-variable REEs, num to be split: ", num_split_single)

        # 2. check violations of bi-variable REE rules
        for idx_bi_ree in range(len(rees_bi_variable)):
            reserved_index = np.where(np.array(exist_conflict) == False)[0]
            if reserved_index.shape[0] == 0:
                break
            ree = rees_bi_variable[idx_bi_ree]
            violate_results = tuples.loc[reserved_index].apply(lambda t_ : violateBiVariableREE_ForTuples(ree, t_, GT), axis=1, result_type='expand')
            violate_tuple_index = np.argwhere(violate_results[0].values == True)
            for idx in violate_tuple_index:
                exist_conflict[reserved_index[idx[0]]] = True

        num_split_bi = np.sum(np.array(exist_conflict) == True)
        print("after using bi-variable REEs, num to be split: ", num_split_bi)

    if useMc is True:
        # 3. check violations of Mc rules
        for Mc_rule in Mc:
            reserved_index = np.where(np.array(exist_conflict) == False)[0]
            if reserved_index.shape[0] == 0:
                break
            violate_results = tuples.loc[reserved_index].apply(lambda t_: violateSingleRule(Mc_rule, t_), axis=1, result_type='expand')
            violate_tuple_index = np.argwhere(violate_results[0].values == True)
            for idx in violate_tuple_index:
                exist_conflict[reserved_index[idx[0]]] = True
                conflicting_attributes[reserved_index[idx[0]]] = violate_results[1].iloc[idx[0]]

        num_split_Mc = np.sum(np.array(exist_conflict) == True)
        print("after using Mc rules, num to be split: ", num_split_Mc)

    return exist_conflict, conflicting_attributes


# split one tuple to two tuples - HyperCube - old tuples without ||
def decideTS_simplified_HyperCube(GT, rees, Mc, tuples, useREE, useMc, output_exist_conflict, Mc_model, check_sharp, Mc_conf, use_Mc_rules, useDittoRules, thr_ditto_rule):
    conflicting_attributes = np.array([None] * tuples.shape[0])
    exist_conflict = np.array([0] * tuples.shape[0])

    check_indices = None
    if useDittoRules is True:
        ditto_not_satisfy_indices = tuples.loc[tuples["confidence"] < thr_ditto_rule].index.values
        check_indices = ditto_not_satisfy_indices
    else:
        check_indices = np.array(range(tuples.shape[0]))

    # if tuples.index[0] in check_indices:  # track
    #     print("\n...... DS-2 Was this tuple inferred as positive data by ditto rules: False")  # track
    # else:  # track
    #     print("\n...... DS-2 Was this tuple inferred as positive data by ditto rules: True")  # track

    # check ## conflict - new schema
    sharp_not_equal_indices = None
    if check_sharp is True:
        sharp_not_equal_indices = []
        for attr in tuples.columns.values:
            reserved_indices = np.where(exist_conflict == 0)[0]
            # reserved_indices = np.array([i for i in reserved_indices if i in check_indices])
            reserved_indices = np.intersect1d(reserved_indices, check_indices)
            if reserved_indices.shape[0] == 0:
                break
            tuples_check = tuples.loc[reserved_indices]
            contain_sharp_indices = tuples_check.loc[tuples_check[attr].astype(str).str.contains("##", na=False)].index.values
            if contain_sharp_indices.shape[0] == 0:
                continue
            series_check0 = tuples.loc[contain_sharp_indices, attr].map(lambda x: x.split("##")[0]).replace("", np.nan)
            series_check1 = tuples.loc[contain_sharp_indices, attr].map(lambda x: x.split("##")[1]).replace("", np.nan)
            whether_equal = (series_check0 == series_check1)
            not_equal_indices = whether_equal.loc[whether_equal == False].index.values
            if not_equal_indices.shape[0] > 0:
                exist_conflict[not_equal_indices] = 1
                conflicting_attributes[not_equal_indices] = attr

            sharp_not_equal_indices = sharp_not_equal_indices + list(not_equal_indices)
        sharp_not_equal_indices = np.unique(np.array(sharp_not_equal_indices))

        # num_conflict_sharp = np.sum(exist_conflict == 1)
        # print("after checking sharp symbol, num of conflict tuples: ", num_conflict_sharp)

    if useREE is True:
        rees_single_variable, rees_bi_variable = [], []
        for ree in rees:
            if ree.get_tuple_variable_cnt() == 1:
                rees_single_variable.append(ree)
            else:
                rees_bi_variable.append(ree)

        # 1. check violations of single-variable REE rules
        # method-1
        for ree in rees_single_variable:
            reserved_indices = np.where(exist_conflict == 0)[0]
            # reserved_indices = np.array([i for i in reserved_indices if i in check_indices])
            reserved_indices = np.intersect1d(reserved_indices, check_indices)
            if reserved_indices.shape[0] == 0:  # there's no left tuples to check, and all of them need to be split
                break
            violate, violate_by_attr, violate_rule_indices = violateSingleRule_ForTuples(ree, tuples.loc[reserved_indices])
            if violate is True:
                # for index_t_ in violate_rule_indices:
                #     exist_conflict[index_t_] = 1
                #     conflicting_attributes[index_t_] = violate_by_attr
                exist_conflict[np.array(violate_rule_indices)] = 1
                conflicting_attributes[np.array(violate_rule_indices)] = violate_by_attr
                # print("\n...... DS-3 this tuple was detected conflicting by a single-variable REE:\n", ree.print_rule())  # track
        # method-2
        # for index_t_, t_ in tuples.iterrows():
        #     for ree in rees_single_variable:
        #         violate, violate_by_attr = violateSingleRule(ree, t_)
        #         if violate is True:
        #             exist_conflict[index_t_] = 1
        #             conflicting_attributes[index_t_] = violate_by_attr
        #             break
        # num_conflict_single = np.sum(exist_conflict == 1)
        # print("after using single-variable REEs, num of conflict tuples: ", num_conflict_single)
        # f = open(output_exist_conflict + "/exist_conflict_singleREE.txt", "w")
        # f.writelines(str(exist_conflict.tolist()))
        # f.close()

        # 2. check violations of bi-variable REE rules (t0.A = t1.A), fix tuples as t0 and GT as t1
        for ree in rees_bi_variable:
            reserved_indices = np.where(exist_conflict == 0)[0]
            # reserved_indices = np.array([i for i in reserved_indices if i in check_indices])
            reserved_indices = np.intersect1d(reserved_indices, check_indices)
            if reserved_indices.shape[0] == 0:  # there's no left tuples to check, and all of them need to be split
                break
            rhs = ree.get_RHS()
            attr_RHS = rhs.get_attr1()
            if rhs.get_type() == "constant" and rhs.get_index1() == 1:
                continue
            value2tid_tuples, value2tid_GT = {}, {}
            constants_in_X = [[], []]
            # 1. get key
            key_attributes_non_constant = []
            for predicate in ree.get_currents():
                if predicate.get_type() == "non-constant":
                    key_attributes_non_constant.append(predicate.get_attr1())
                else:
                    tid = predicate.get_index1()
                    constants_in_X[tid].append((predicate.get_attr1(), predicate.get_constant()))  # (A, a)
            # 2. get value of tuples
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
                value2tid_tuples[value] = df  # df.index.values
            if len(value2tid_tuples) == 0:  # there's no tuples satisfy the non-constant predicates in X of current ree; we should go for the next ree
                continue
            # 3. get value of GT
            # (1) first filter tuples that not satisfy the constant predicates
            reserved_indices_GT = GT.index
            for attr, v in constants_in_X[1]:  # constant predicates
                tuples_check_constants = GT.loc[reserved_indices_GT]
                reserved_indices_GT = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
                if reserved_indices_GT.shape[0] == 0:
                    break
            if reserved_indices_GT.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                continue
            # (2) then construct dict by non-constant predicates
            for value, df in GT.loc[reserved_indices_GT].groupby(key_attributes_non_constant):
                value2tid_GT[value] = df  # df.index.values
            if len(value2tid_GT) == 0:  # there's no tuples satisfy the non-constant predicates in X of current ree; we should go for the next ree
                continue
            # 4. check Y
            violate_indices = []
            # (1) check constant Y
            if rhs.get_type() == "constant":
                for key in value2tid_tuples.keys():
                    if key not in value2tid_GT.keys():
                        continue
                    df = value2tid_tuples[key]
                    null_indices = df.loc[df[attr_RHS].isnull()].index.values
                    violateY_indices = df.loc[df[attr_RHS].astype(str) != rhs.get_constant()].index.values
                    # violateY_indices = np.array([i for i in violateY_indices if i not in null_indices])
                    violateY_indices = np.array(list(set(violateY_indices).difference(set(null_indices))))
                    if violateY_indices.shape[0] > 0:
                        violate_indices.append(violateY_indices)
                        # print("\n...... DS-4 this tuple was detected conflicting by a bi-variable REE:\n", ree.print_rule())  # track
                for indices in violate_indices:
                    # for index in indices:
                    #     exist_conflict[index] = 1
                    #     conflicting_attributes[index] = attr_RHS
                    exist_conflict[indices] = 1
                    conflicting_attributes[indices] = attr_RHS
            # (2) check non-constant Y
            else:
                for key in value2tid_tuples.keys():
                    if key not in value2tid_GT.keys():
                        continue
                    check_attr = attr_RHS
                    check_series = value2tid_tuples[key][check_attr]
                    for index_t_ in check_series.index:
                        value_in_tuples = value2tid_tuples[key].loc[index_t_][check_attr]
                        if pd.isnull(value_in_tuples) is True:  # None can not be regarded as violate the rules
                            continue
                        if value2tid_GT[key].loc[value2tid_GT[key][check_attr].astype(str) != str(value_in_tuples)].index.values.shape[0] != 0:  # as long as there exists one violation, then the tuple should be split
                            violate_indices.append(index_t_)
                            # print("\n...... DS-4 this tuple was detected conflicting by a bi-variable REE:\n", ree.print_rule())  # track
                # for index in violate_indices:
                #     exist_conflict[index] = 1
                #     conflicting_attributes[index] = attr_RHS
                if len(violate_indices) > 0:
                    exist_conflict[np.array(violate_indices)] = 1
                    conflicting_attributes[np.array(violate_indices)] = attr_RHS

        # num_conflict_bi = np.sum(exist_conflict == 1)
        # print("after using bi-variable REEs, num of conflict tuples: ", num_conflict_bi)
        # f = open(output_exist_conflict + "/exist_conflict_biREE.txt", "w")
        # f.writelines(str(exist_conflict.tolist()))
        # f.close()

    if useMc is True:
        if use_Mc_rules is True:
            # 3. check violations of Mc rules
            for Mc_rule in Mc:
                reserved_index = np.where(exist_conflict == 0)[0]
                # reserved_index = np.array([i for i in reserved_index if i in check_indices])
                reserved_index = np.intersect1d(reserved_index, check_indices)
                if reserved_index.shape[0] == 0:
                    break
                # violate_results = tuples.loc[reserved_index].apply(lambda t_: violateSingleRule(Mc_rule, t_), axis=1, result_type='expand')
                # violate_tuple_index = np.argwhere(violate_results[0].values == True)
                bar_A = Mc_rule.get_currents().get_attr1()
                B = Mc_rule.get_currents().get_attr2()
                barAB = bar_A + [B]
                scores = Mc_model.predictMcScore_new(tuples.loc[reserved_index, barAB])
                scores = [float(i) for i in scores]  # cast type
                # violate_tuple_index = np.argwhere(np.array(scores) >= Mc_rule.get_currents().get_confidence())
                violate_tuple_index = np.argwhere(np.array(scores) >= Mc_conf)
                if violate_tuple_index.shape[0] == 0:
                    continue
                for idx in violate_tuple_index:
                    exist_conflict[reserved_index[idx[0]]] = 1
                    # conflicting_attributes[reserved_index[idx[0]]] = violate_results[1].iloc[idx[0]]
                    conflicting_attributes[reserved_index[idx[0]]] = Mc_rule.get_currents().get_attr2()
                    # print("\n...... DS-5 this tuple was detected conflicting by a Mc rule with real confidence " + str(Mc_conf) + ":\n", Mc_rule.print_rule())  # track
        else:
            reserved_index = np.where(exist_conflict == 0)[0]
            # reserved_index = np.array([i for i in reserved_index if i in check_indices])
            reserved_index = np.intersect1d(reserved_index, check_indices)
            # mc_scores = pd.DataFrame(columns=["scores"], index=reserved_index)
            if reserved_index.shape[0] != 0:
                scores = Mc_model.predictMcScore_new(tuples.loc[reserved_index])
                scores = [float(i) for i in scores]
                # mc_scores.loc[reserved_index, "scores"] = np.array(scores)
                # mc_scores.to_csv(output_exist_conflict + "/DS_mc_scores.csv")
                violate_tuple_index = np.argwhere(np.array(scores) >= Mc_conf)
                if violate_tuple_index.shape[0] > 0:
                    # for idx in violate_tuple_index:
                    #     exist_conflict[reserved_index[idx[0]]] = 1
                    violate_tuple_index = np.array([reserved_index[idx[0]] for idx in violate_tuple_index])
                    exist_conflict[violate_tuple_index] = 1
                    # print("\n...... DS-5 this tuple was detected conflicting by Mc model with confidence " + str(Mc_conf))  # track

        # num_conflict_Mc = np.sum(exist_conflict == 1)
        # print("after using Mc rules, num of conflict tuples: ", num_conflict_Mc)

    # f = open(output_exist_conflict + "/exist_conflict.txt", "w")
    # f.writelines(str(exist_conflict.tolist()))
    # f.close()

    return exist_conflict, conflicting_attributes, sharp_not_equal_indices


def decideTS_simplified_HyperCube_count_violations(GT, rees, Mc, tuples, Mc_model, Mc_conf, use_Mc_rules, useDittoRules, thr_ditto_rule):
    # count violations for ree rules
    arr = np.zeros((tuples.shape[0], tuples.shape[1]))
    violation_matrix_ree = pd.DataFrame(arr, columns=tuples.columns.values).astype(int)
    violation_matrix_ree.index = range(tuples.shape[0])
    # count violations for mc rules
    violation_matrix_mc = pd.DataFrame(arr, columns=tuples.columns.values).astype(int)
    violation_matrix_mc.index = range(tuples.shape[0])
    # count violations for mc model
    violation_series_mc = pd.Series([0] * tuples.shape[0]).astype(int)
    violation_series_mc.index = range(tuples.shape[0])

    if useDittoRules is True:
        ditto_not_satisfy_indices = tuples.loc[tuples["confidence"] < thr_ditto_rule].index.values
        check_indices = ditto_not_satisfy_indices
    else:
        check_indices = np.array(range(tuples.shape[0]))
    if check_indices.shape[0] == 0:
        print("After checking ditto rules, there is no left tuples to detect violations")
        return violation_matrix_ree, violation_matrix_mc, violation_series_mc

    rees_single_variable, rees_bi_variable = [], []
    for ree in rees:
        if ree.get_tuple_variable_cnt() == 1:
            rees_single_variable.append(ree)
        else:
            rees_bi_variable.append(ree)

    # 1. check violations of single-variable REE rules
    for ree in rees_single_variable:
        violate, violate_by_attr, violate_rule_indices = violateSingleRule_ForTuples(ree, tuples.loc[check_indices])
        if violate is True:
            violation_matrix_ree.loc[violate_rule_indices, violate_by_attr] += 1

    # 2. check violations of bi-variable REE rules (t0.A = t1.A), fix tuples as t0 and GT as t1
    for ree in rees_bi_variable:
        reserved_indices = copy.deepcopy(check_indices)
        rhs = ree.get_RHS()
        attr_RHS = rhs.get_attr1()
        if rhs.get_type() == "constant" and rhs.get_index1() == 1:
            continue
        value2tid_tuples, value2tid_GT = {}, {}
        constants_in_X = [[], []]
        # (1). get key
        key_attributes_non_constant = []
        for predicate in ree.get_currents():
            if predicate.get_type() == "non-constant":
                key_attributes_non_constant.append(predicate.get_attr1())
            else:
                tid = predicate.get_index1()
                constants_in_X[tid].append((predicate.get_attr1(), predicate.get_constant()))  # (A, a)
        # (2). get value of tuples
        # (a) first filter tuples that not satisfy the constant predicates
        for attr, v in constants_in_X[0]:  # constant predicates
            tuples_check_constants = tuples.loc[reserved_indices]
            reserved_indices = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
            if reserved_indices.shape[0] == 0:
                break
        if reserved_indices.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
            continue
        # (b) then construct dict by non-constant predicates
        for value, df in tuples.loc[reserved_indices].groupby(key_attributes_non_constant):
            value2tid_tuples[value] = df  # df.index.values
        if len(value2tid_tuples) == 0:  # there's no tuples satisfy the non-constant predicates in X of current ree; we should go for the next ree
            continue
        # (3). get value of GT
        # (a) first filter tuples that not satisfy the constant predicates
        reserved_indices_GT = GT.index
        for attr, v in constants_in_X[1]:  # constant predicates
            tuples_check_constants = GT.loc[reserved_indices_GT]
            reserved_indices_GT = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
            if reserved_indices_GT.shape[0] == 0:
                break
        if reserved_indices_GT.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
            continue
        # (b) then construct dict by non-constant predicates
        for value, df in GT.loc[reserved_indices_GT].groupby(key_attributes_non_constant):
            value2tid_GT[value] = df  # df.index.values
        if len(value2tid_GT) == 0:  # there's no tuples satisfy the non-constant predicates in X of current ree; we should go for the next ree
            continue
        # (4). check Y
        violate_indices = []
        # (a) check constant Y
        if rhs.get_type() == "constant":
            for key in value2tid_tuples.keys():
                if key not in value2tid_GT.keys():
                    continue
                df = value2tid_tuples[key]
                null_indices = df.loc[df[attr_RHS].isnull()].index.values
                violateY_indices = df.loc[df[attr_RHS].astype(str) != rhs.get_constant()].index.values
                violateY_indices = np.array(list(set(violateY_indices).difference(set(null_indices))))
                if violateY_indices.shape[0] > 0:
                    # violation_matrix_ree.loc[violateY_indices, attr_RHS] += 1
                    violation_matrix_ree.loc[violateY_indices, attr_RHS] += value2tid_GT[key].shape[0]  # violate with different tuples in Gamma
        # (b) check non-constant Y
        else:
            for key in value2tid_tuples.keys():
                if key not in value2tid_GT.keys():
                    continue
                check_attr = attr_RHS
                check_series = value2tid_tuples[key][check_attr]
                for index_t_ in check_series.index:
                    value_in_tuples = value2tid_tuples[key].loc[index_t_][check_attr]
                    if pd.isnull(value_in_tuples) is True:
                        continue
                    not_equal_num = value2tid_GT[key].loc[value2tid_GT[key][check_attr].astype(str) != str(value_in_tuples)].index.values.shape[0]  # as long as there exists one violation, then the tuple should be split
                    if not_equal_num != 0:
                        # violation_matrix_ree.loc[index_t_, attr_RHS] += 1
                        violation_matrix_ree.loc[index_t_, attr_RHS] += not_equal_num  # violate with different tuples in Gamma

    # 3. check violations by Mc
    if use_Mc_rules is True:
        for Mc_rule in Mc:
            bar_A = Mc_rule.get_currents().get_attr1()
            B = Mc_rule.get_currents().get_attr2()
            barAB = bar_A + [B]
            scores = Mc_model.predictMcScore_new(tuples.loc[check_indices, barAB])
            scores = [float(i) for i in scores]  # cast type
            violate_tuple_index = np.argwhere(np.array(scores) >= Mc_conf)
            if violate_tuple_index.shape[0] == 0:
                continue
            for idx in violate_tuple_index:
                violation_matrix_mc.loc[check_indices[idx[0]], Mc_rule.get_currents().get_attr2()] += 1
    else:
        scores = Mc_model.predictMcScore_new(tuples.loc[check_indices])
        scores = [float(i) for i in scores]
        violate_tuple_index = np.argwhere(np.array(scores) >= Mc_conf)
        if violate_tuple_index.shape[0] > 0:
            violate_tuple_index = np.array([check_indices[idx[0]] for idx in violate_tuple_index])  # not know the violated attributes
            violation_series_mc.loc[violate_tuple_index] += 1

    return violation_matrix_ree, violation_matrix_mc, violation_series_mc


def splitTuples_initial(tuples):
    tuples_initial = [None, None]
    for tid in [0, 1]:
        tuples_initial[tid] = copy.deepcopy(tuples)

    all_attrs = tuples.columns.values
    for attr in all_attrs:
        split_indices = tuples.loc[tuples[attr].astype(str).str.contains("\|\|", na=False)].index.values
        for tid in [0, 1]:
            tuples_initial[tid].loc[split_indices, attr] = tuples.loc[split_indices, attr].map(lambda x: x.split("||")[tid]).replace("", np.nan)
    return tuples_initial


# split one tuple to two tuples - HyperCube - new schema of tuples with ||
def decideTS_simplified_HyperCube_new(GT, rees, Mc, tuples, useREE, useMc, output_exist_conflict, Mc_model, check_sharp, Mc_conf, use_Mc_rules, evaluateDS_syn, useDittoRules, thr_ditto_rule, output_results):
    tuples_to_check = splitTuples_initial(tuples)

    exist_conflict_t0_t1 = [None, None]
    sharp_not_equal_indices = [None, None]
    for tid in [0, 1]:
        # print("\n...... DS-1 initial split into tuple t" + str(tid) + ":\n", tuples_to_check[tid])  # track
        exist_conflict_t0_t1[tid], no_use, sharp_not_equal_indices[tid] = decideTS_simplified_HyperCube(GT, rees, Mc, tuples_to_check[tid], useREE, useMc, output_exist_conflict, Mc_model, check_sharp, Mc_conf, use_Mc_rules, useDittoRules, thr_ditto_rule)
        # print("\n...... DS-6 after DecideTS, whether this tuple t" + str(tid) + " was detected conflicting is: " + str(exist_conflict_t0_t1[tid][0]))  # track

    # only when two tuples both checked to be True; then the original tuples should be split, i.e., return True
    if evaluateDS_syn is True:
        exist_conflict = exist_conflict_t0_t1[0] * exist_conflict_t0_t1[1]  # and
    else:
        exist_conflict = exist_conflict_t0_t1[0] + exist_conflict_t0_t1[1]  # or
        index = np.where(exist_conflict == 2)[0]
        exist_conflict[index] = 1

    # print("\n...... DS-partial result: this tracked_tid tuple was detected conflicting as: " + str(exist_conflict[0]))  # track

    sharp_split_indices = None
    if sharp_not_equal_indices[0] is None:
        if sharp_not_equal_indices[1] is not None:
            sharp_split_indices = sharp_not_equal_indices[1]
    else:
        if sharp_not_equal_indices[1] is None:
            sharp_split_indices = sharp_not_equal_indices[0]
        else:
            sharp_split_indices = np.array(list(set(sharp_not_equal_indices[0]).union(set(sharp_not_equal_indices[1]))))

    if output_results is True:
        f = open(output_exist_conflict + "partial_exist_conflict.txt", "w")
        f.writelines(str(exist_conflict.tolist()))
        f.close()

        if sharp_split_indices is not None:
            f = open(output_exist_conflict + "sharp_split_indices.txt", "w")
            f.writelines(str(sharp_split_indices.tolist()))
            f.close()

    return exist_conflict, no_use, sharp_split_indices


def decideTS_simplified_HyperCube_new_count_violations(GT, rees, Mc, tuples, Mc_model, Mc_conf, use_Mc_rules, useDittoRules, thr_ditto_rule, labels):
    tuples_to_check = splitTuples_initial(tuples)

    violation_matrix_ree = [None, None]  # count violations for ree rules
    violation_matrix_mc = [None, None]  # count violations for mc rules
    violation_series_mc = [None, None]  # count violations for mc models
    for tid in [0, 1]:
        violation_matrix_ree[tid], violation_matrix_mc[tid], violation_series_mc[tid] = decideTS_simplified_HyperCube_count_violations(GT, rees, Mc, tuples_to_check[tid], Mc_model, Mc_conf, use_Mc_rules, useDittoRules, thr_ditto_rule)

    split_indices = np.where(labels == 1)[0]
    repair_indices = np.where(labels == -1)[0]
    positive_indices = np.where(labels == 0)[0]

    # ------------------------- strategy-1: add all violations
    violation_matrix_ree_total = violation_matrix_ree[0] + violation_matrix_ree[1]
    violation_matrix_mc_total = violation_matrix_mc[0] + violation_matrix_mc[1]
    violation_series_mc_total = violation_series_mc[0] + violation_series_mc[1]

    num_vio_ree = violation_matrix_ree_total.sum().sum()
    num_vio_ree_split = violation_matrix_ree_total.loc[split_indices].sum().sum()
    num_vio_ree_repair = violation_matrix_ree_total.loc[repair_indices].sum().sum()
    num_vio_ree_positive = violation_matrix_ree_total.loc[positive_indices].sum().sum()
    if use_Mc_rules:
        num_vio_mc = violation_matrix_mc_total.sum().sum()
        num_vio_mc_split = violation_matrix_mc_total.loc[split_indices].sum().sum()
        num_vio_mc_repair = violation_matrix_mc_total.loc[repair_indices].sum().sum()
        num_vio_mc_positive = violation_matrix_mc_total.loc[positive_indices].sum().sum()
    else:
        num_vio_mc = violation_series_mc_total.sum()
        num_vio_mc_split = violation_series_mc_total.loc[split_indices].sum().sum()
        num_vio_mc_repair = violation_series_mc_total.loc[repair_indices].sum().sum()
        num_vio_mc_positive = violation_series_mc_total.loc[positive_indices].sum().sum()

    print("------------------------ for DecideTS ------------------------")
    print("#### Strategy-1: Add all violations for each cell. In DecideTS:")
    print("In total: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree, num_vio_mc))
    print("For tuples to be split: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree_split, num_vio_mc_split))
    print("For tuples to be repaired: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree_repair, num_vio_mc_repair))
    print("For positive tuples: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree_positive, num_vio_mc_positive))

    # ------------------------- strategy-2: count the max number of violations for each cell
    all_attrs = tuples.columns.values
    arr = np.zeros((tuples.shape[0], tuples.shape[1]))
    violation_matrix_ree_max = pd.DataFrame(arr, columns=all_attrs).astype(int)
    violation_matrix_ree_max.index = range(tuples.shape[0])

    violation_matrix_mc_max = pd.DataFrame(arr, columns=all_attrs).astype(int)
    violation_matrix_mc_max.index = range(tuples.shape[0])

    for attr in all_attrs:
        compare = violation_matrix_ree[0][attr] > violation_matrix_ree[1][attr]
        index_0 = compare.loc[compare == True].index.values
        index_1 = compare.loc[compare == False].index.values
        violation_matrix_ree_max.loc[index_0, attr] = violation_matrix_ree[0].loc[index_0, attr]
        violation_matrix_ree_max.loc[index_1, attr] = violation_matrix_ree[1].loc[index_1, attr]

        if use_Mc_rules:
            compare = violation_matrix_mc[0][attr] > violation_matrix_mc[1][attr]
            index_0 = compare.loc[compare == True].index.values
            index_1 = compare.loc[compare == False].index.values
            violation_matrix_mc_max.loc[index_0, attr] = violation_matrix_mc[0].loc[index_0, attr]
            violation_matrix_mc_max.loc[index_1, attr] = violation_matrix_mc[1].loc[index_1, attr]

    violation_series_mc_max = violation_series_mc[0] + violation_series_mc[1]

    num_vio_ree = violation_matrix_ree_max.sum().sum()
    num_vio_ree_split = violation_matrix_ree_max.loc[split_indices].sum().sum()
    num_vio_ree_repair = violation_matrix_ree_max.loc[repair_indices].sum().sum()
    num_vio_ree_positive = violation_matrix_ree_max.loc[positive_indices].sum().sum()
    if use_Mc_rules:
        num_vio_mc = violation_matrix_mc_max.sum().sum()
        num_vio_mc_split = violation_matrix_mc_max.loc[split_indices].sum().sum()
        num_vio_mc_repair = violation_matrix_mc_max.loc[repair_indices].sum().sum()
        num_vio_mc_positive = violation_matrix_mc_max.loc[positive_indices].sum().sum()
    else:
        num_vio_mc = violation_series_mc_max.sum()
        num_vio_mc_split = violation_series_mc_max.loc[split_indices].sum()
        num_vio_mc_repair = violation_series_mc_max.loc[repair_indices].sum()
        num_vio_mc_positive = violation_series_mc_max.loc[positive_indices].sum()

    print("------------------------ for DecideTS ------------------------")
    print("#### Strategy-2: Count the max number of violations for each cell")
    print("In total: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree, num_vio_mc))
    print("For tuples to be split: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree_split, num_vio_mc_split))
    print("For tuples to be repaired: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree_repair, num_vio_mc_repair))
    print("For positive tuples: there are {} and {} violations caught by REEs and Mc".format(num_vio_ree_positive, num_vio_mc_positive))


# this function is not right, because as long as the tuple satisfy the X of REE, then we can use Y to impute values; no need to use Y only for violation
def attributesAssignment_REEs(rees, GT, new_tuples, t_, attr_assign):
    success = False
    value_assign = t_[attr_assign]

    finish_impute = [False, False]
    impute_values = [None, None]

    # use REEs rules to check whether to assign values to t1 or t2
    for ree in rees:
        rhs = ree.get_RHS()

        if rhs.get_attr1() != attr_assign:
            continue

        if finish_impute[0] is True and finish_impute[1] is True:
            break

        # (1) single-variable REEs
        if ree.get_tuple_variable_cnt() == 1:
            for tid in [0, 1]:  # for tuple t0 and t1
                if finish_impute[tid] is True:
                    continue
                violate, no_use = violateSingleRule(ree, new_tuples.iloc[tid])
                if violate is True and not pd.isnull(rhs.get_constant()):  # satisfy X but not Y
                    impute_values[tid] = rhs.get_constant()
                    finish_impute[tid] = True

        # (2) bi-variable REEs
        elif ree.get_tuple_variable_cnt() == 2:
            # to check whether tuple can be regarded as t0 or t1
            satisfy_cpredicates = [None, None]
            satisfy_cpredicates[0] = satisfyConstantPredicatesInX(ree, new_tuples.iloc[0])
            satisfy_cpredicates[1] = satisfyConstantPredicatesInX(ree, new_tuples.iloc[1])

            for tid in [0, 1]:  # for new_tuples[0] and new_tuples[1]
                if finish_impute[tid] is True:
                    continue
                satisfy_consPre = satisfy_cpredicates[tid]
                if satisfy_consPre[0] is False and satisfy_consPre[1] is False:
                    continue
                for index, row in GT.iterrows():
                    if finish_impute[tid] is True:
                        break
                    if rhs.get_type() == "non-constant":
                        violate1, violate2 = False, False
                        if satisfy_consPre[0] is True:  # tuple can be regarded as t0
                            violate1, no_use = violateBiVariableREE(ree, new_tuples.iloc[tid], row)
                        if satisfy_consPre[1] is True and violate1 is False:  # tuple can be regarded as t1
                            violate2, no_use = violateBiVariableREE(ree, row, new_tuples.iloc[tid])
                        if (violate1 is True or violate2 is True) and not pd.isnull(row[attr_assign]):
                            impute_values[tid] = row[attr_assign]
                            finish_impute[tid] = True
                    elif rhs.get_type() == "constant":
                        violate = False
                        if rhs.get_index1() == 0 and satisfy_consPre[0] is True:  # tuple can be regarded as t0
                            violate, no_use = violateBiVariableREE(ree, new_tuples.iloc[tid], row)
                        elif rhs.get_index1() == 1 and satisfy_consPre[1] is True:  # tuple can be regarded as t1
                            violate, no_use = violateBiVariableREE(ree, row, new_tuples.iloc[tid])
                        if violate is True and not pd.isnull(rhs.get_constant()):  # satisfy X but not Y
                            impute_values[tid] = rhs.get_constant()
                            finish_impute[tid] = True

    # assign value to t1 or t2
    if impute_values[0] == value_assign and impute_values[1] != value_assign:
        success = True
        new_tuples.iloc[0][attr_assign] = value_assign
        if impute_values[1] is not None:
            new_tuples.iloc[1][attr_assign] = impute_values[1]

    elif impute_values[0] != value_assign and impute_values[1] == value_assign:
        success = True
        new_tuples.iloc[1][attr_assign] = value_assign
        if impute_values[0] is not None:
            new_tuples.iloc[0][attr_assign] = impute_values[0]

    return success, new_tuples, finish_impute, impute_values


# new_tuples contains two rows, i.e., two new tuples
def attributesAssignment_Mc(Mc, new_tuples, t_, attr_assign):
    value_assign = t_[attr_assign]
    attrs_finished = []
    # for attrs in new_tuples.loc[:, (new_tuples != "nan").any()].columns:
    #     attrs_finished.append(attrs)
    for attr in new_tuples.columns:
        if not new_tuples[attr].isnull().all():
            attrs_finished.append(attr)

    barA_B = new_tuples[attrs_finished + [attr_assign]]
    barA_B.iloc[0][attr_assign] = value_assign
    barA_B.iloc[1][attr_assign] = value_assign
    scores = predictMcScore(barA_B, predictor, wikidata_embed, wikidata_dic, item_dict, item_dict_reverse)

    confidence = 0.5  # if no Mc rules to be applied, then set confidence to be 0.5
    for Mc_rule in Mc:
        predicate = Mc_rule.get_currents()
        A_attr_list = predicate.get_attr1()
        B_attr = predicate.get_attr2()
        if set(attrs_finished) != set(A_attr_list):
            continue
        if attr_assign != B_attr:
            continue
        confidence = predicate.get_confidence()
        break

    successfully_assign = False
    for i in range(len(scores)):
        if scores[i] < confidence:
            new_tuples.iloc[i][attr_assign] = value_assign
            successfully_assign = True

    if successfully_assign is False:
        if scores[0] < scores[1]:
            new_tuples.iloc[0][attr_assign] = value_assign
        else:
            new_tuples.iloc[1][attr_assign] = value_assign

    return True, new_tuples


def attributesAssignment_Mc_new(Mc, new_tuple0, new_tuple1, t_, attr_assign):
    t = [None, None]
    t[0] = new_tuple0.to_frame().transpose()  # only one line
    t[1] = new_tuple1.to_frame().transpose()  # only one line

    value_assign = t_[attr_assign]
    attrs_finished = [[], []]
    for attr in t[0].columns:
        for tid in [0, 1]:
            if not pd.isnull(t[tid].iloc[0][attr]):  # only one line
                attrs_finished[tid].append(attr)

    assign_tid = []
    scores = []
    for tid in [0, 1]:
        barA_B = t[tid][attrs_finished[tid] + [attr_assign]]
        barA_B.iloc[0][attr_assign] = value_assign
        score = predictMcScore(barA_B, predictor, wikidata_embed, wikidata_dic, item_dict, item_dict_reverse)[0]
        scores.append(score)

        confidence = 0.5  # if no Mc rules to be applied, then set confidence to be 0.5
        for Mc_rule in Mc:
            predicate = Mc_rule.get_currents()
            A_attr_list = predicate.get_attr1()
            B_attr = predicate.get_attr2()
            if set(attrs_finished[tid]) != set(A_attr_list):
                continue
            if attr_assign != B_attr:
                continue
            confidence = predicate.get_confidence()
            break

        if score < confidence:
            assign_tid.append(tid)

    if len(assign_tid) == 0:
        if scores[0] < scores[1]:
            assign_tid.append(0)
        else:
            assign_tid.append(1)

    return assign_tid


# tuples.loc[both_inequal_indices][assign_attr] do not contains "##"
def attributesAssignment_Mc_batch(Mc, new_tuples, tuples, both_inequal_indices, assign_attr, Mc_model, Mc_conf, tuple_status):
    tuples_ = tuples.loc[both_inequal_indices]
    assign_values = tuples_[assign_attr]
    all_indices = tuples_.index

    new_tuples_ = [None, None]
    new_tuples_[0] = new_tuples[0].loc[both_inequal_indices]
    new_tuples_[1] = new_tuples[1].loc[both_inequal_indices]

    attrs_finished = [[], []]
    for tid in [0, 1]:
        for attr in new_tuples_[tid].columns:
            if attr == assign_attr:
                continue
            if not new_tuples_[0][attr].isnull().all():
                attrs_finished[tid].append(attr)
    # attrs_finished_both = [i for i in attrs_finished[0] if i in attrs_finished[1]]
    attrs_finished_both = np.intersect1d(np.array(attrs_finished[0]), np.array(attrs_finished[1])).tolist()

    no_common_attrs_flag = False
    if len(attrs_finished_both) == 0:
        no_common_attrs_flag = True

    confidence = [Mc_conf, Mc_conf]
    # for tid in [0, 1]:
    #     for Mc_rule in Mc:
    #         predicate = Mc_rule.get_currents()
    #         A_attr_list = predicate.get_attr1()
    #         B_attr = predicate.get_attr2()
    #         if no_common_attrs_flag is False:
    #             if set(attrs_finished_both) != set(A_attr_list):
    #                 continue
    #         else:
    #             if set(attrs_finished[tid]) != set(A_attr_list):
    #                 continue
    #         if assign_attr != B_attr:
    #             continue
    #         confidence[tid] = predicate.get_confidence()
    #         break
    #     if no_common_attrs_flag is False:
    #         confidence[1] = confidence[0]
    #         break

    scores = [None, None]
    for tid in [0, 1]:
        if no_common_attrs_flag is False:
            barA_B = new_tuples_[tid][attrs_finished_both + [assign_attr]]
        else:
            barA_B = new_tuples_[tid][attrs_finished[tid] + [assign_attr]]
        barA_B = barA_B.copy()
        barA_B.loc[:, assign_attr] = assign_values
        # scores[tid] = predictMcScore(barA_B, predictor, wikidata_embed, wikidata_dic, item_dict, item_dict_reverse)
        scores[tid] = Mc_model.predictMcScore_new(barA_B)
        # scores[tid] = [float(i) for i in scores[tid]]  # cast type
        scores[tid] = list(map(float, scores[tid]))  # cast type

        indices = np.where(np.array(scores[tid]) < confidence[tid])[0]
        new_tuples_[tid].loc[all_indices[indices], assign_attr] = assign_values.loc[all_indices[indices]]
        tuple_status[tid].loc[all_indices[indices], assign_attr] = 1  # assign

    # in case that the score are all higher than confidence, then we assign the value the tuples with less score
    # whether_assign_t0 = [True if scores[0][i] < scores[1][i] else False for i in range(len(scores[0]))]
    whether_assign_t0 = (np.array(scores[0]) < np.array(scores[1]))
    assign_t0_indices = np.where(whether_assign_t0 == True)[0]
    assign_t1_indices = np.where(whether_assign_t0 == False)[0]
    new_tuples_[0].loc[all_indices[assign_t0_indices], assign_attr] = assign_values.loc[all_indices[assign_t0_indices]]
    new_tuples_[1].loc[all_indices[assign_t1_indices], assign_attr] = assign_values.loc[all_indices[assign_t1_indices]]
    tuple_status[0].loc[all_indices[assign_t0_indices], assign_attr] = 1  # assign
    tuple_status[1].loc[all_indices[assign_t1_indices], assign_attr] = 1  # assign

    return new_tuples_[0], new_tuples_[1], tuple_status


# tuples.loc[null_indices][assign_attr] do contains "##"
def attributesAssignment_Mc_batch_sharp(new_tuples, tuples, null_indices, assign_attr, Mc_model):
    tuples_ = tuples.loc[null_indices]
    assign_values_left = tuples_[assign_attr].map(lambda x: x.split("##")[0])  # value in the left
    assign_values_right = tuples_[assign_attr].map(lambda x: x.split("##")[1])  # value in the left

    new_tuples_ = [None, None]
    new_tuples_[0] = new_tuples[0].loc[null_indices]
    new_tuples_[1] = new_tuples[1].loc[null_indices]

    attrs_finished = [[], []]
    for tid in [0, 1]:
        for attr in new_tuples_[tid].columns:
            if attr == assign_attr:
                continue
            if not new_tuples_[0][attr].isnull().all():
                attrs_finished[tid].append(attr)
    # attrs_finished_both = [i for i in attrs_finished[0] if i in attrs_finished[1]]
    attrs_finished_both = np.intersect1d(np.array(attrs_finished[0]), np.array(attrs_finished[1])).tolist()

    no_common_attrs_flag = False
    if len(attrs_finished_both) == 0:
        no_common_attrs_flag = True

    scores = [None, None, None, None]  # t0-c0, t0-c1, t1-c0, t1-c1
    k = 0
    for tid in [0, 1]:
        if no_common_attrs_flag is False:
            barA_B = new_tuples_[tid][attrs_finished_both + [assign_attr]]
        else:
            barA_B = new_tuples_[tid][attrs_finished[tid] + [assign_attr]]
        barA_B = barA_B.copy()
        barA_B.loc[:, assign_attr] = assign_values_left
        scores[k] = Mc_model.predictMcScore_new(barA_B)
        k = k + 1

        barA_B.loc[:, assign_attr] = assign_values_right
        scores[k] = Mc_model.predictMcScore_new(barA_B)
        k = k + 1

    scores = pd.DataFrame(scores).transpose().astype(float)
    scores.columns = ["0_0", "0_1", "1_0", "1_1"]
    scores.index = null_indices

    # the columns idx with minimal values for each row
    max_idx = scores.idxmin(axis=1)
    indices_0_0 = max_idx.loc[max_idx == "0_0"].index.values
    indices_0_1 = max_idx.loc[max_idx == "0_1"].index.values
    indices_1_0 = max_idx.loc[max_idx == "1_0"].index.values
    indices_1_1 = max_idx.loc[max_idx == "1_1"].index.values

    # assign the values with minimal scores
    new_tuples_[0].loc[indices_0_0, assign_attr] = assign_values_left.loc[indices_0_0]
    new_tuples_[0].loc[indices_0_1, assign_attr] = assign_values_right.loc[indices_0_1]
    new_tuples_[1].loc[indices_1_0, assign_attr] = assign_values_left.loc[indices_1_0]
    new_tuples_[1].loc[indices_1_1, assign_attr] = assign_values_right.loc[indices_1_1]

    # assign the remaining values
    new_tuples_[1].loc[indices_0_0, assign_attr] = assign_values_right.loc[indices_0_0]
    new_tuples_[1].loc[indices_0_1, assign_attr] = assign_values_left.loc[indices_0_1]
    new_tuples_[0].loc[indices_1_0, assign_attr] = assign_values_right.loc[indices_1_0]
    new_tuples_[0].loc[indices_1_1, assign_attr] = assign_values_left.loc[indices_1_1]

    return new_tuples_[0], new_tuples_[1]


# split mis-matched tuples and re-assign attributes
def split(rees, Mc, GT, tuples, exist_conflict, conflicting_attributes, main_attribute, output_path, useREE, useMc):
    all_attrs = tuples.columns.values.tolist()
    all_new_tuples = []
    for i in range(tuples.shape[0]):
        if exist_conflict[i] is False:
            all_new_tuples.append(tuples.iloc[i])
            all_new_tuples.append(tuples.iloc[i])
            continue
        t_ = tuples.iloc[i]
        splitting_attr = conflicting_attributes[i]
        reserved_attrs = [i for i in all_attrs]

        # 1. generate new tuples by splitting t_
        new_tuples = np.full((2, len(all_attrs)), None)
        new_tuples = pd.DataFrame(new_tuples, columns=all_attrs)
        main_attrs = main_attribute.split("||")
        for main_attr in main_attrs:
            new_tuples.iloc[0][main_attr] = t_[main_attr]
            new_tuples.iloc[1][main_attr] = t_[main_attr]
            reserved_attrs.remove(main_attr)

        # 2. assign values of splitting attribute to two new tuples
        # if "||" in t_[splitting_attr]:
        #     values = t_[splitting_attr].split("||")
        #     new_tuples.iloc[0][splitting_attr] = values[0].strip()
        #     new_tuples.iloc[1][splitting_attr] = values[1].strip()
        # else:  # may not right!
        #     new_tuples.iloc[0][splitting_attr] = t_[splitting_attr]  # assign to tuple 0; the other one, i.e., tuple 1 be nan
        # reserved_attrs.remove(splitting_attr)

        # 3. re-assign the rest attribute values
        # (1) assign the values that reserved for tuples before merging
        attrs_contain = []
        for attr in reserved_attrs:
            if "||" in str(t_[attr]):
                values = t_[attr].split("||")
                new_tuples.iloc[0][attr] = values[0].strip()
                new_tuples.iloc[1][attr] = values[1].strip()
                attrs_contain.append(attr)
        reserved_attrs = [i for i in reserved_attrs if i not in attrs_contain]

        # (2) assign the rest single values to t1 or t2 by using REEs and Mc rules
        for attr_assign in reserved_attrs:
            success_ree, success_mc, finish_impute, impute_values = False, False, [False, False], [None, None]
            if useREE is True:
                # a. use REEs rules
                success_ree, new_tuples, finish_impute, impute_values = attributesAssignment_REEs(rees, GT, new_tuples, t_, attr_assign)
                if success_ree is True:
                    continue
                if useMc is False:
                    # cannot use REEs to assign value and not use Mc to decide, then randomly assign the value to tuple t0
                    new_tuples.iloc[0][attr_assign] = t_[attr_assign]
                    if impute_values[1] is not None:
                        new_tuples.iloc[1][attr_assign] = impute_values[1]
            if useMc is True:
                # b. use Mc rules
                success_mc, new_tuples = attributesAssignment_Mc(Mc, new_tuples, t_, attr_assign)
                # c. imputation by REEs
                if pd.isnull(new_tuples.iloc[0][attr_assign]) and impute_values[0] is not None:
                    new_tuples.iloc[0][attr_assign] = impute_values[0]
                if pd.isnull(new_tuples.iloc[1][attr_assign]) and impute_values[1] is not None:
                    new_tuples.iloc[1][attr_assign] = impute_values[1]

        all_new_tuples.append(new_tuples.iloc[0])
        all_new_tuples.append(new_tuples.iloc[1])

    # output new tuples to file
    split_tuples = pd.DataFrame(all_new_tuples)
    split_tuples.to_csv(output_path, index=False)

    return split_tuples


# split mis-matched tuples and re-assign attributes, using REEs by HyperCube
def split_HyperCube(rees, Mc, GT, tuples, exist_conflict, exist_conflict_groundtruth, sharp_split_indices, output_path, useREE, useMc, Mc_model, Mc_conf, her, evaluate_separately, tuple_status, whether_output_results):
    predicted_conflict_indices = np.where(exist_conflict == 1)[0]  # conflict: split and repair
    groundtruth_split_indices = np.where(exist_conflict_groundtruth == 1)[0]  # only split
    groundtruth_repair_indices = np.where(exist_conflict_groundtruth == -1)[0]  # only repair

    reserved_all_indices = np.array(list(set(predicted_conflict_indices).union(set(groundtruth_split_indices))))
    if reserved_all_indices.shape[0] == 0:
        return None, None, None, None, None

    # generate two tables for the new split tuples
    all_attrs = tuples.columns.values.tolist()
    new_tuples = [None, None]
    for tid in [0, 1]:
        new_tuples[tid] = np.full((tuples.shape[0], len(all_attrs)), None)
        new_tuples[tid] = pd.DataFrame(new_tuples[tid], columns=all_attrs)

    # assign main attributes and values that reserved for two tuples
    # main_attrs = main_attribute.split("||")
    # for tid in [0, 1]:
        # new_tuples[tid][main_attrs] = tuples[main_attrs]
        # new_tuples[tid]["id"] = tuples["id"].map(lambda x: x.split("||")[tid])
    for attr in all_attrs:
        split_indices = tuples.loc[tuples[attr].astype(str).str.contains("\|\|", na=False)].index.values
        new_tuples[0].loc[split_indices, attr] = tuples.loc[split_indices, attr].map(lambda x: x.split("||")[0]).replace("", np.nan)
        new_tuples[1].loc[split_indices, attr] = tuples.loc[split_indices, attr].map(lambda x: x.split("||")[1]).replace("", np.nan)

    # directly assign tuples that has no conflicts
    # no_conflict_index = np.where(exist_conflict == 0)[0]  # original version(only split not repair)
    no_conflict_index = np.array(list(set(range(tuples.shape[0])).difference(set(reserved_all_indices))))  # not all included
    if no_conflict_index.shape[0] != 0:
        # indices_key = indices2ids.keys()
        # indices_key_null = np.array([i for i in no_conflict_index if i not in indices_key])
        # indices_key_change = np.array([i for i in no_conflict_index if i in indices_key])
        # ids_value_change = np.array([indices2ids[i] for i in indices_key_change])
        for tid in [0, 1]:
            new_tuples[tid].loc[no_conflict_index, :] = tuples.loc[no_conflict_index, :]
            # new_tuples[tid].loc[indices_key_change, "id"] = ids_value_change
            # new_tuples[tid].loc[indices_key_null, "id"] = np.nan

    # print("\n...... AA-0, current initially split tuples:")  # track
    # print("\n...... AA-0, t0:\n", new_tuples[0])  # track
    # print("\n...... AA-0, t1:\n", new_tuples[1])  # track

    # assign and impute each attribute one by one, and record status for each attribute of each tuple
    # reserved_attrs = [i for i in all_attrs if i not in main_attrs]
    # reserved_attrs.remove("id")
    reserved_attrs = [i for i in all_attrs if i != "id"]
    imputation_attr_value_indices = [{}, {}]  # record the imputation information by REEs;  imputation_attr_value_indices[tid][attr][value] = array([index1, ..., indexn])
    initial_null_indices_all = {}
    for assign_attr in reserved_attrs:
        assign_values = tuples[assign_attr]

        # if str(assign_values.iloc[0]) == "nan" or assign_values.iloc[0] is None or str(assign_values.iloc[0]) == "":  # track
        #     continue  # track
        # print("\n...... AA-1, current assign_attr: [ " + assign_attr + "] , and assign_value: [ " + assign_values + " ]")  # track

        tuples_check0 = new_tuples[0].loc[reserved_all_indices]
        tuples_check1 = new_tuples[1].loc[reserved_all_indices]
        initial_null_indices0 = tuples_check0.loc[tuples_check0[assign_attr].isnull()].index.values
        initial_null_indices1 = tuples_check1.loc[tuples_check1[assign_attr].isnull()].index.values
        # initial_null_indices = [i for i in initial_null_indices0 if i in initial_null_indices1]  # both null means need to assign values; if one has value and the other is null, means only need to do imputation
        initial_null_indices = np.intersect1d(initial_null_indices0, initial_null_indices1)  # both null means need to assign values; if one has value and the other is null, means only need to do imputation
        initial_null_indices_all[assign_attr] = initial_null_indices

        if useREE is True:
            useful_single_rees, useful_bi_rees = filter_useful_rees(rees, assign_attr)
            for tid in [0, 1]:  # for each table of new tuples
                flag_all_filled = False
                # 1. use single-variable REEs
                for ree in useful_single_rees:
                    tuples_check = new_tuples[tid].loc[reserved_all_indices]
                    reserved_indices = tuples_check.loc[tuples_check[assign_attr].isnull()].index.values
                    if reserved_indices.shape[0] == 0:
                        flag_all_filled = True
                        break
                    for predicate in ree.get_currents():
                        attr = predicate.get_attr1()
                        constant = predicate.get_constant()
                        reserved_tuples_check = new_tuples[tid].loc[reserved_indices]
                        reserved_indices = reserved_tuples_check.loc[reserved_tuples_check[attr].astype(str) == constant].index.values
                        if reserved_indices.shape[0] == 0:
                            break
                    if reserved_indices.shape[0] != 0:  # the indices of tuples that satisfy X of ree
                        rhs = ree.get_RHS()
                        value = rhs.get_constant()
                        if pd.notnull(value):
                            new_tuples[tid].loc[reserved_indices, assign_attr] = value
                            tuple_status[tid].loc[reserved_indices, assign_attr] = 3  # impute
                            # print("\n...... AA-1(impute in advance): tuple t" + str(tid) + ": attribute [ " + assign_attr + " ] was assigned by value [ " + value + " ], by REE: " + ree.print_rule())  # track
                        # if assign_attr not in imputation_attr_value_indices[tid].keys():
                        #     imputation_attr_value_indices[tid][assign_attr] = {}
                        # if value not in imputation_attr_value_indices[tid][assign_attr].keys():
                        #     imputation_attr_value_indices[tid][assign_attr][value] = []
                        # imputation_attr_value_indices[tid][assign_attr][value] += [idx for idx in reserved_indices]
                # 2. use bi-variable REEs
                for ree in useful_bi_rees:
                    if flag_all_filled is True:
                        break
                    rhs = ree.get_RHS()
                    if rhs.get_type() == "constant" and rhs.get_index1() == 1:
                        continue
                    tuples_check = new_tuples[tid].loc[reserved_all_indices]
                    reserved_indices = tuples_check.loc[tuples_check[assign_attr].isnull()].index.values
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
                        tuples_check_constants = new_tuples[tid].loc[reserved_indices]
                        reserved_indices = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
                        if reserved_indices.shape[0] == 0:
                            break
                    if reserved_indices.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                        continue
                    # (2) then construct dict by non-constant predicates
                    for value, df in new_tuples[tid].loc[reserved_indices].groupby(key_attributes_non_constant):
                        value2tid_tuples[value] = df  # df.index.values
                    if len(value2tid_tuples) == 0:
                        continue
                    # get value of GT
                    # (1) first filter tuples that not satisfy the constant predicates
                    reserved_indices_GT = GT.index
                    for attr, v in constants_in_X[1]:  # constant predicates
                        tuples_check_constants = GT.loc[reserved_indices_GT]
                        reserved_indices_GT = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
                        if reserved_indices_GT.shape[0] == 0:
                            break
                    if reserved_indices_GT.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                        continue
                    # (2) then construct dict by non-constant predicates
                    for value, df in GT.loc[reserved_indices_GT].groupby(key_attributes_non_constant):
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
                                new_tuples[tid].loc[indices, assign_attr] = rhs.get_constant()
                                tuple_status[tid].loc[indices, assign_attr] = 3  # impute
                                # print("\n...... AA-1(impute in advance): tuple t" + str(tid) + ": attribute [ " + assign_attr + " ] was assigned by value [ " + rhs.get_constant() + " ], by REE: " + ree.print_rule())  # track
                            # if assign_attr not in imputation_attr_value_indices[tid].keys():
                            #     imputation_attr_value_indices[tid][assign_attr] = {}
                            # if rhs.get_constant() not in imputation_attr_value_indices[tid][assign_attr].keys():
                            #     imputation_attr_value_indices[tid][assign_attr][rhs.get_constant()] = []
                            # imputation_attr_value_indices[tid][assign_attr][rhs.get_constant()] += [idx for idx in indices]
                    # apply non-constant Y
                    else:
                        for key in value2tid_tuples.keys():
                            if key not in value2tid_GT.keys():
                                continue
                            indices = value2tid_tuples[key].index.values
                            series_GT = value2tid_GT[key][assign_attr]
                            valid_idx = series_GT.first_valid_index()
                            value = series_GT.loc[valid_idx] if valid_idx is not None else None  # choose the first non-null value for imputation!  can be changed to obtain all values for imputation
                            if pd.notnull(value):
                                new_tuples[tid].loc[indices, assign_attr] = value
                                tuple_status[tid].loc[indices, assign_attr] = 3  # impute
                                # print("\n...... AA-1(impute in advance): tuple t" + str(tid) + ": attribute [ " + assign_attr + " ] was assigned by value [ " + value + " ], by REE: " + ree.print_rule())  # track
                            # if assign_attr not in imputation_attr_value_indices[tid].keys():
                            #     imputation_attr_value_indices[tid][assign_attr] = {}
                            # if value not in imputation_attr_value_indices[tid][assign_attr].keys():
                            #     imputation_attr_value_indices[tid][assign_attr][value] = []
                            # imputation_attr_value_indices[tid][assign_attr][value] += [idx for idx in indices]

        # values assignment has higher priority than imputation by rules; so if two tuples both not assigned the values, then we use Mc to decide given value to which tuple
        if useMc is True:
            if len(initial_null_indices) == 0:
                continue

            # for new schema with || and ##
            two_value_indices = assign_values.loc[assign_values.astype(str).str.contains("##", na=False)].index.values
            one_value_indices = np.array(list(set(assign_values.index).difference(set(two_value_indices))))
            # (1) deal with only one value
            one_value_indices_check = np.intersect1d(initial_null_indices, one_value_indices)
            inequal_value_indices_0 = np.where((new_tuples[0].loc[one_value_indices_check][assign_attr] == assign_values.loc[one_value_indices_check]).values == False)[0]
            inequal_value_indices_1 = np.where((new_tuples[1].loc[one_value_indices_check][assign_attr] == assign_values.loc[one_value_indices_check]).values == False)[0]
            both_inequal_indices = np.intersect1d(inequal_value_indices_0, inequal_value_indices_1)
            # both_inequal_indices = np.array([one_value_indices_check[i] for i in both_inequal_indices])
            both_inequal_indices = one_value_indices_check[both_inequal_indices]
            if both_inequal_indices.shape[0] != 0:
                new_tuples0, new_tuples1, tuple_status = attributesAssignment_Mc_batch(Mc, new_tuples, tuples, both_inequal_indices, assign_attr, Mc_model, Mc_conf, tuple_status)
                new_tuples[0].loc[both_inequal_indices, assign_attr] = new_tuples0.loc[:, assign_attr]
                new_tuples[1].loc[both_inequal_indices, assign_attr] = new_tuples1.loc[:, assign_attr]
                # if str(new_tuples0.loc[0, assign_attr]) != "nan" and new_tuples0.loc[0, assign_attr] is not None and str(new_tuples0.loc[0, assign_attr]) != "":  # track
                    # print("\n...... AA-3 (After impute by REEs, use Mc to assign the rest values. Note that assignment has higher priority than imputation here.)")  # track
                    # print("\n...... AA-3-1(deal with one value), tuple t0: attribute [ " + assign_attr + " ] was assigned by value [ " + new_tuples0.loc[:, assign_attr] + " ], by Mc model")  # track
                # if str(new_tuples1.loc[0, assign_attr]) != "nan" and new_tuples1.loc[0, assign_attr] is not None and str(new_tuples1.loc[0, assign_attr]) != "":  # track
                    # print("\n...... AA-3-1(deal with one value), tuple t1: attribute [ " + assign_attr + " ] was assigned by value [ " + new_tuples1.loc[:, assign_attr] + " ], by Mc model")  # track

            # (2) deal with two values split by ##, directly use Mc results to replace REE results, because the two values to be assigned must be true, only not known assign which to which
            two_value_indices_check = np.intersect1d(initial_null_indices, two_value_indices)
            if two_value_indices_check.shape[0] != 0:
                new_tuples0, new_tuples1 = attributesAssignment_Mc_batch_sharp(new_tuples, tuples, two_value_indices_check, assign_attr, Mc_model)
                new_tuples[0].loc[two_value_indices_check, assign_attr] = new_tuples0.loc[:, assign_attr]
                new_tuples[1].loc[two_value_indices_check, assign_attr] = new_tuples1.loc[:, assign_attr]
                tuple_status[0].loc[two_value_indices_check, assign_attr] = 1  # assign
                tuple_status[1].loc[two_value_indices_check, assign_attr] = 1  # assign
                # if str(new_tuples0.loc[0, assign_attr]) != "nan" and new_tuples0.loc[0, assign_attr] is not None and str(new_tuples0.loc[0, assign_attr]) != "":  # track
                    # print("\n...... AA-3 (After impute by REEs, use Mc to assign the rest values. Note that assignment has higher priority than imputation here.)")  # track
                    # print("\n...... AA-3-2(deal with two values split by ##), tuple t0: attribute [ " + assign_attr + " ] was assigned by value [ " + new_tuples0.loc[:, assign_attr] + " ], by Mc model")  # track
                # if str(new_tuples1.loc[0, assign_attr]) != "nan" and new_tuples1.loc[0, assign_attr] is not None and str(new_tuples1.loc[0, assign_attr]) != "":  # track
                    # print("\n...... AA-3-2(deal with two values split by ##), tuple t1: attribute [ " + assign_attr + " ] was assigned by value [ " + new_tuples1.loc[:, assign_attr] + " ], by Mc model")  # track

            # for old schema with || but not with ##
            # inequal_value_indices_0 = np.where((new_tuples[0].loc[initial_null_indices][assign_attr] == assign_values.loc[initial_null_indices]).values == False)[0]
            # inequal_value_indices_1 = np.where((new_tuples[1].loc[initial_null_indices][assign_attr] == assign_values.loc[initial_null_indices]).values == False)[0]
            # both_inequal_indices = np.array([i for i in inequal_value_indices_0 if i in inequal_value_indices_1])
            # both_inequal_indices = np.array([initial_null_indices[i] for i in both_inequal_indices])
            # # b. use Mc rules
            # if both_inequal_indices.shape[0] != 0:
            #     new_tuples0, new_tuples1 = attributesAssignment_Mc_batch(Mc, new_tuples, tuples, both_inequal_indices, assign_attr, Mc_model)
            #     new_tuples[0].loc[both_inequal_indices, assign_attr] = new_tuples0.loc[:, assign_attr]
            #     new_tuples[1].loc[both_inequal_indices, assign_attr] = new_tuples1.loc[:, assign_attr]
            #     # for index in both_inequal_indices:
            #     #     assign_tid = attributesAssignment_Mc_new(Mc, new_tuples[0].loc[index], new_tuples[1].loc[index], tuples.loc[index], assign_attr)
            #     #     for tid in assign_tid:
            #     #         # impute_value_by_ree = new_tuples[tid].loc[index][assign_attr]
            #     #         # if not pd.isnull(impute_value_by_ree):
            #     #         #     imputation_attr_value_indices[tid][assign_attr][impute_value_by_ree].remove(index)
            #     #         # assign values to tid
            #     #         new_tuples[tid].loc[index][assign_attr] = assign_values[index]
            #     #         # update imputation_attr_value_indices
            #     #         # if assign_attr not in imputation_attr_value_indices[tid].keys():
            #     #         #     imputation_attr_value_indices[tid][assign_attr] = {}
            #     #         # if assign_values[index] not in imputation_attr_value_indices[tid][assign_attr].keys():
            #     #         #     imputation_attr_value_indices[tid][assign_attr][assign_values[index]] = []
            #     #         # imputation_attr_value_indices[tid][assign_attr][assign_values[index]].append(index)

    # distinguish tuples should be split or repaired, and return the final results for DS
    start_distinguish = timeit.default_timer()
    final_exist_conflict_DS = distinguish_split_repair(new_tuples, exist_conflict, sharp_split_indices, her, tuple_status, tuples)
    end_distinguish = timeit.default_timer()
    extra_decideTS_time = end_distinguish - start_distinguish

    # directly assign tuples that has no conflicts, for the false positive in DS result
    # After this, tuples with groundtruth label -1 and 0 are all directly assigned.
    no_conflict_remain_index = np.array(list(set(predicted_conflict_indices).difference(set(groundtruth_split_indices))))
    if no_conflict_remain_index.shape[0] != 0:
        for tid in [0, 1]:
            new_tuples[tid].loc[no_conflict_remain_index, :] = tuples.loc[no_conflict_remain_index, :]
            tuple_status[tid].loc[no_conflict_remain_index, :] = np.array([None] * len(all_attrs))  # do nothing

    # repair the tuples that should be repaired, i.e., with label -1
    if evaluate_separately is False:
        groundtruth_repair_indices = np.where(final_exist_conflict_DS == -1)[0]
        if groundtruth_repair_indices.shape[0] != 0:
            for tid in [0, 1]:
                new_tuples[tid].loc[groundtruth_repair_indices, :] = tuples.loc[groundtruth_repair_indices, :]
                tuple_status[tid].loc[groundtruth_repair_indices, :] = np.array([None] * len(all_attrs))  # do nothing
    if groundtruth_repair_indices.shape[0] != 0:
        tuples_repaired, tuple_status = repair_tuples(new_tuples[0], groundtruth_repair_indices, useREE, useMc, rees, GT, Mc_model, Mc_conf, tuple_status)
        new_tuples[0].loc[groundtruth_repair_indices] = tuples_repaired
        new_tuples[1].loc[groundtruth_repair_indices] = tuples_repaired
        # print("\n...... AA-5, if the tracked tuple is checked as repair rather than split")  # track
        # print("\n...... AA-5, after repairing, tuple t0 and t1 (same): \n", tuples_repaired)  # track

    # randomly assign values that can not decided by REEs if useMc is False; values assignment has higher priority than imputation by rules
    # if evaluate_separately is False:
    #     groundtruth_split_indices = np.where(final_exist_conflict_DS == 1)[0]
    # if useMc is False:
    #     for assign_attr in reserved_attrs:
    #         assign_values = tuples[assign_attr]
    #
    #         initial_null_indices = np.intersect1d(initial_null_indices_all[assign_attr], groundtruth_split_indices)  # both null means need to assign values; if one has value and the other is null, means only need to do imputation
    #
    #         if len(initial_null_indices) == 0:
    #             continue
    #         two_value_indices = assign_values.loc[assign_values.astype(str).str.contains("##", na=False)].index.values
    #         one_value_indices = np.array(list(set(assign_values.index).difference(set(two_value_indices))))
    #         # (1) deal with one value
    #         one_value_indices_check = np.intersect1d(initial_null_indices, one_value_indices)
    #         inequal_value_indices_0 = np.where((new_tuples[0].loc[one_value_indices_check][assign_attr] == assign_values.loc[one_value_indices_check]).values == False)[0]
    #         inequal_value_indices_1 = np.where((new_tuples[1].loc[one_value_indices_check][assign_attr] == assign_values.loc[one_value_indices_check]).values == False)[0]
    #         both_inequal_indices = np.intersect1d(inequal_value_indices_0, inequal_value_indices_1)
    #         # both_inequal_indices = np.array([one_value_indices_check[i] for i in both_inequal_indices])
    #         both_inequal_indices = one_value_indices_check[both_inequal_indices]
    #
    #         random_assign_idx = np.random.randint(0, 2, size=both_inequal_indices.shape[0])
    #         assign_t0_indices = both_inequal_indices[np.where(random_assign_idx == 0)[0]]
    #         assign_t1_indices = both_inequal_indices[np.where(random_assign_idx == 1)[0]]
    #
    #         new_tuples[0].loc[assign_t0_indices, assign_attr] = assign_values.loc[assign_t0_indices]
    #         new_tuples[1].loc[assign_t1_indices, assign_attr] = assign_values.loc[assign_t1_indices]
    #         tuple_status[0].loc[assign_t0_indices, assign_attr] = 1  # assign
    #         tuple_status[1].loc[assign_t1_indices, assign_attr] = 1  # assign
    #
    #         # (2) deal with two values separated by ##
    #         two_value_indices_check = np.intersect1d(initial_null_indices, two_value_indices)
    #         assign_values_left = assign_values.loc[two_value_indices_check].map(lambda x: x.split("##")[0])  # value in the left
    #         assign_values_right = assign_values.loc[two_value_indices_check].map(lambda x: x.split("##")[1])  # value in the left
    #
    #         # a. if value_left(or value_right) has been assigned to t0(or t1), then we assign value_right(or value_left) to t1(or t0)
    #         equal_value_indices_0_left = np.where((new_tuples[0].loc[two_value_indices_check][assign_attr] == assign_values_left.loc[two_value_indices_check]).values == True)[0]
    #         equal_value_indices_0_right = np.where((new_tuples[0].loc[two_value_indices_check][assign_attr] == assign_values_right.loc[two_value_indices_check]).values == True)[0]
    #         equal_value_indices_1_left = np.where((new_tuples[1].loc[two_value_indices_check][assign_attr] == assign_values_left.loc[two_value_indices_check]).values == True)[0]
    #         equal_value_indices_1_right = np.where((new_tuples[1].loc[two_value_indices_check][assign_attr] == assign_values_right.loc[two_value_indices_check]).values == True)[0]
    #
    #         # equal_value_indices_0_left = np.array([two_value_indices_check[i] for i in equal_value_indices_0_left])
    #         # equal_value_indices_0_right = np.array([two_value_indices_check[i] for i in equal_value_indices_0_right])
    #         # equal_value_indices_1_left = np.array([two_value_indices_check[i] for i in equal_value_indices_1_left])
    #         # equal_value_indices_1_right = np.array([two_value_indices_check[i] for i in equal_value_indices_1_right])
    #         #  |
    #         # \/
    #
    #         equal_value_indices_0_left = two_value_indices_check[equal_value_indices_0_left]
    #         equal_value_indices_0_right = two_value_indices_check[equal_value_indices_0_right]
    #         equal_value_indices_1_left = two_value_indices_check[equal_value_indices_1_left]
    #         equal_value_indices_1_right = two_value_indices_check[equal_value_indices_1_right]
    #
    #         new_tuples[1].loc[equal_value_indices_0_left, assign_attr] = assign_values_right.loc[equal_value_indices_0_left]
    #         new_tuples[1].loc[equal_value_indices_0_right, assign_attr] = assign_values_left.loc[equal_value_indices_0_right]
    #         new_tuples[0].loc[equal_value_indices_1_left, assign_attr] = assign_values_right.loc[equal_value_indices_1_left]
    #         new_tuples[0].loc[equal_value_indices_1_right, assign_attr] = assign_values_left.loc[equal_value_indices_1_right]
    #
    #         tuple_status[1].loc[equal_value_indices_0_left, assign_attr] = 1  # assign
    #         tuple_status[1].loc[equal_value_indices_0_right, assign_attr] = 1  # assign
    #         tuple_status[0].loc[equal_value_indices_1_left, assign_attr] = 1  # assign
    #         tuple_status[0].loc[equal_value_indices_1_right, assign_attr] = 1  # assign
    #
    #         # b. randomly assign value_left and value_right to t0 or t1, each has one value
    #         inequal_value_indices_0_left = np.where((new_tuples[0].loc[two_value_indices_check][assign_attr] == assign_values_left.loc[two_value_indices_check]).values == False)[0]
    #         inequal_value_indices_0_right = np.where((new_tuples[0].loc[two_value_indices_check][assign_attr] == assign_values_right.loc[two_value_indices_check]).values == False)[0]
    #         inequal_value_indices_0 = np.intersect1d(inequal_value_indices_0_left, inequal_value_indices_0_right)
    #         # inequal_value_indices_0 = np.array([two_value_indices_check[i] for i in inequal_value_indices_0])
    #         inequal_value_indices_0 = two_value_indices_check[inequal_value_indices_0]
    #
    #         inequal_value_indices_1_left = np.where((new_tuples[1].loc[two_value_indices_check][assign_attr] == assign_values_left.loc[two_value_indices_check]).values == False)[0]
    #         inequal_value_indices_1_right = np.where((new_tuples[1].loc[two_value_indices_check][assign_attr] == assign_values_right.loc[two_value_indices_check]).values == False)[0]
    #         inequal_value_indices_1 = np.intersect1d(inequal_value_indices_1_left, inequal_value_indices_1_right)
    #         # inequal_value_indices_1 = np.array([two_value_indices_check[i] for i in inequal_value_indices_1])
    #         inequal_value_indices_1 = two_value_indices_check[inequal_value_indices_1]
    #
    #         # inequal_value_indices = np.array([i for i in inequal_value_indices_0 if i in inequal_value_indices_1])  # the indices of t0 and t1, which both not be assigned value_left or value_right
    #         inequal_value_indices = np.intersect1d(inequal_value_indices_0, inequal_value_indices_1)  # the indices of t0 and t1, which both not be assigned value_left or value_right
    #
    #         random_assign_t0_idx = np.random.randint(0, 2, size=inequal_value_indices.shape[0])
    #         random_assign_t1_idx = np.ones(inequal_value_indices.shape[0]) - random_assign_t0_idx  # complementary idx
    #
    #         assign_t0_left_indices = inequal_value_indices[np.where(random_assign_t0_idx == 0)[0]]
    #         assign_t0_right_indices = inequal_value_indices[np.where(random_assign_t0_idx == 1)[0]]
    #         assign_t1_left_indices = inequal_value_indices[np.where(random_assign_t1_idx == 0)[0]]
    #         assign_t1_right_indices = inequal_value_indices[np.where(random_assign_t1_idx == 1)[0]]
    #
    #         new_tuples[0].loc[assign_t0_left_indices, assign_attr] = assign_values_left.loc[assign_t0_left_indices]
    #         new_tuples[0].loc[assign_t0_right_indices, assign_attr] = assign_values_right.loc[assign_t0_right_indices]
    #         new_tuples[1].loc[assign_t1_left_indices, assign_attr] = assign_values_left.loc[assign_t1_left_indices]
    #         new_tuples[1].loc[assign_t1_right_indices, assign_attr] = assign_values_right.loc[assign_t1_right_indices]
    #
    #         tuple_status[0].loc[assign_t0_left_indices, assign_attr] = 1  # assign
    #         tuple_status[0].loc[assign_t0_right_indices, assign_attr] = 1  # assign
    #         tuple_status[1].loc[assign_t1_left_indices, assign_attr] = 1  # assign
    #         tuple_status[1].loc[assign_t1_right_indices, assign_attr] = 1  # assign

    # save AA results to files
    if whether_output_results is True:
        new_tuples[0].to_csv(output_path + "tuples_splitting_t0.csv", index=False)
        new_tuples[1].to_csv(output_path + "tuples_splitting_t1.csv", index=False)

    # print("\n...... AA-final results, tuple t0: \n", new_tuples[0])  # track
    # print("\n...... AA-final results, tuple t1: \n", new_tuples[1])  # track

    return new_tuples, imputation_attr_value_indices, final_exist_conflict_DS, extra_decideTS_time, tuple_status


# distinguish whether to split or repair a conflict tuple by referencing KG;
# In exist_conflict, replace "1" to "-1" for tuples that need to be repaired rather than split
def distinguish_split_repair(conflict_tuples, exist_conflict, sharp_split_indices, her, tuple_status, origin_tuples):
    '''
    1⃣ both t1 and t2 can be matched to entity in KG:
        (1) Both t1 and t2 match more than one entity: split;
        (2) One of them can match more than one entity, while the other only match one entity: split;
        (3) Both of them can only match one entity: (a) the id of matched entities are not equal: split; (b) Otherwise: repair;
    2⃣ One of t1 and t2 cannot match the entities in KG, while the other can be matched successfully:
        (1) The success one match more than one entity: repair;
        (2) The success one match only one entity: repair;
    3⃣ Neither t1 nor t2 can match the entities in KG: repair;
    '''
    conflict_indices = np.where(exist_conflict == 1)[0]  # conflict: split and repair
    if sharp_split_indices is not None:
        conflict_indices = np.array(list(set(conflict_indices).difference(set(sharp_split_indices))))  # no need to check the tuples of sharp_split_indices, which must belong to split rather than repair

    # # Recover the values to be None that has been imputed during AA
    # temp_df = copy.deepcopy(conflict_tuples)  # obtain the tuples after been assigned but before been imputed
    # for tid in [0, 1]:
    #     for attr in origin_tuples.columns.values:
    #         impute_indices = tuple_status[tid].loc[tuple_status[tid][attr] == 3].index.values
    #         temp_df[tid].loc[impute_indices, attr] = None

    # another strategy: only use main attributes with ||
    origin_tuples_check = origin_tuples.loc[conflict_indices]
    all_attrs = origin_tuples_check.columns.values.tolist()
    temp_df = [None, None]
    for tid in [0, 1]:
        temp_df[tid] = np.full((origin_tuples_check.shape[0], len(all_attrs)), None)
        temp_df[tid] = pd.DataFrame(temp_df[tid], columns=all_attrs)
        temp_df[tid].index = conflict_indices
    for attr in all_attrs:
        split_indices = origin_tuples_check.loc[origin_tuples_check[attr].astype(str).str.contains("\|\|", na=False)].index.values
        temp_df[0].loc[split_indices, attr] = origin_tuples_check.loc[split_indices, attr].map(lambda x: x.split("||")[0]).replace("", np.nan)
        temp_df[1].loc[split_indices, attr] = origin_tuples_check.loc[split_indices, attr].map(lambda x: x.split("||")[1]).replace("", np.nan)

    no_match_indices = [None, None]
    one_match_indices = [None, None]
    # isMatch = [None, None]
    # matched_entity_ids = [None, None]
    # oneMatch = [None, None]
    # multiMatch = [None, None]
    # for tid in [0, 1]:
    #     isMatch[tid], matched_entity_ids[tid], oneMatch[tid], multiMatch[tid] = her.reference_KG_set_intersection_filter_candidates(temp_df[tid].loc[conflict_indices])
    #     no_match_indices[tid] = isMatch[tid].loc[isMatch[tid] == False].index.values
    #     one_match_indices[tid] = oneMatch[tid].loc[oneMatch[tid] == True].index.values

    temp_df_check = pd.concat([temp_df[0].loc[conflict_indices], temp_df[1].loc[conflict_indices]])
    temp_df_check.index = range(temp_df_check.shape[0])
    start = timeit.default_timer()
    isMatch_, matched_entity_ids_, oneMatch_, multiMatch_ = her.reference_KG_set_intersection_filter_candidates(temp_df_check)
    end = timeit.default_timer()
    print("reference_KG_set_intersection_filter_candidates time:", end - start)

    isMatch = np.array_split(isMatch_, 2)
    matched_entity_ids = np.array_split(matched_entity_ids_, 2)
    oneMatch = np.array_split(oneMatch_, 2)
    multiMatch = np.array_split(multiMatch_, 2)
    for tid in [0, 1]:
        isMatch[tid].index = conflict_indices
        matched_entity_ids[tid].index = conflict_indices
        oneMatch[tid].index = conflict_indices
        multiMatch[tid].index = conflict_indices

    for tid in [0, 1]:
        no_match_indices[tid] = isMatch[tid].loc[isMatch[tid] == False].index.values
        one_match_indices[tid] = oneMatch[tid].loc[oneMatch[tid] == True].index.values

    # print("\n...... AA-4(distinguish_split_repair)")  # track
    # print("\n...... AA-4, tuple t0, isMatch:" + str(isMatch[0].iloc[0]) + ", oneMatch:" + str(oneMatch[0].iloc[0]) + ", multiMatch:" + str(multiMatch[0].iloc[0]))  # track
    # print("\n...... AA-4, tuple t1, isMatch:" + str(isMatch[1].iloc[0]) + ", oneMatch:" + str(oneMatch[1].iloc[0]) + ", multiMatch:" + str(multiMatch[1].iloc[0]))  # track

    no_match_indices_either = np.array(list(set(no_match_indices[0]).union(set(no_match_indices[1]))))
    if no_match_indices_either.shape[0] > 0:
        exist_conflict[no_match_indices_either] = -1  # repair
        # print("\n...... AA-4, one of t0 and t1 has no matches in KG, so it should be repaired.")  # track

    both_one_match_indices = np.array(list(set(one_match_indices[0]).intersection(set(one_match_indices[1]))))
    df_temp = pd.concat([matched_entity_ids[0].loc[both_one_match_indices], matched_entity_ids[1].loc[both_one_match_indices]], axis=1)
    df_temp.columns = ["eid_t0", "eid_t1"]
    df_temp["equal"] = np.where(df_temp["eid_t0"] == df_temp["eid_t1"], True, False)
    one_equal_match_indices = df_temp.loc[df_temp["equal"] == True].index.values
    if one_equal_match_indices.shape[0] > 0:
        exist_conflict[one_equal_match_indices] = -1  # repair
        # print("\n...... AA-4, both t0 and t1 have one match in KG, and the match is the same, so it should be repaired.")  # track

    # print("\n...... AA-4, for the tracked tuple, final label in exist_conflict: " + str(exist_conflict[0]))  # track

    # print("=== conflict indices:", conflict_indices.shape[0])
    # print("=== no_match_indices[0]:", no_match_indices[0].shape[0])
    # print("=== no_match_indices[1]:", no_match_indices[1].shape[0])
    # print("=== no_match_indices_either:", no_match_indices_either.shape[0])
    # print("=== one_equal_match_indices:", one_equal_match_indices.shape[0])

    return exist_conflict


def repair_tuples(tuples, repair_indices, useREE, useMc, rees, GT, Mc_model, Mc_conf, tuple_status):
    tuples_to_repair = tuples.loc[repair_indices]
    all_attrs = tuples_to_repair.columns.values.tolist()

    tuples_repaired = np.full((tuples_to_repair.shape[0], len(all_attrs)), None)
    tuples_repaired = pd.DataFrame(tuples_repaired, columns=all_attrs)
    tuples_repaired.index = repair_indices

    basic_df = np.full((tuples_to_repair.shape[0], len(all_attrs)), None)
    basic_df = pd.DataFrame(basic_df, columns=all_attrs)
    basic_df.index = repair_indices

    remain_attr_indices = {}
    for attr in all_attrs:
        split_indices = tuples_to_repair.loc[tuples_to_repair[attr].astype(str).str.contains("\|\|", na=False)].index.values
        tuples_repaired.loc[split_indices, attr] = tuples_to_repair.loc[split_indices, attr].map(lambda x: x.split("||")[1]).replace("", np.nan)

        contain_sharp_indices = tuples_to_repair.loc[tuples_to_repair[attr].astype(str).str.contains("##", na=False)].index.values
        tuples_repaired.loc[contain_sharp_indices, attr] = tuples_to_repair.loc[contain_sharp_indices, attr].map(lambda x: x.split("##")[1]).replace("", np.nan)

        remain_indices = np.array(list(set(repair_indices).difference(set(split_indices)).difference(set(contain_sharp_indices))))
        tuples_repaired.loc[remain_indices, attr] = tuples_to_repair.loc[remain_indices, attr]
        remain_attr_indices[attr] = remain_indices  # record the indices for each single conflict attribute

        # save the basic table that only contains || and ##
        basic_df.loc[split_indices, attr] = tuples_repaired.loc[split_indices, attr]
        basic_df.loc[contain_sharp_indices, attr] = tuples_repaired.loc[contain_sharp_indices, attr]

    reserved_attrs = [i for i in all_attrs if i != "id"]
    if useREE:
        for attr_to_check in reserved_attrs:
            useful_single_rees, useful_bi_rees = filter_useful_rees(rees, attr_to_check)
            # 1. check violations of single-variable REE rules
            for ree in useful_single_rees:
                violate, violate_by_attr, violate_rule_indices = violateSingleRule_ForTuples(ree, tuples_repaired)
                if violate is True:
                    tuples_repaired.loc[violate_rule_indices, violate_by_attr] = None
                    tuple_status[0].loc[violate_rule_indices, violate_by_attr] = 2  # repair
                    tuple_status[1].loc[violate_rule_indices, violate_by_attr] = 2  # repair

            # 2. check violations of bi-variable REE rules (t0.A = t1.A), fix tuples as t0 and GT as t1
            for ree in useful_bi_rees:
                reserved_indices = remain_attr_indices[attr_to_check]
                isnull_indices = tuples_repaired.loc[tuples_repaired[attr_to_check].isnull()].index.values
                reserved_indices = np.array(list(set(reserved_indices).difference(set(isnull_indices))))
                rhs = ree.get_RHS()
                attr_RHS = rhs.get_attr1()
                if rhs.get_type() == "constant" and rhs.get_index1() == 1:
                    continue
                value2tid_tuples, value2tid_GT = {}, {}
                constants_in_X = [[], []]
                # 1. get key
                key_attributes_non_constant = []
                for predicate in ree.get_currents():
                    if predicate.get_type() == "non-constant":
                        key_attributes_non_constant.append(predicate.get_attr1())
                    else:
                        tid = predicate.get_index1()
                        constants_in_X[tid].append((predicate.get_attr1(), predicate.get_constant()))  # (A, a)
                # 2. get value of tuples
                # (1) first filter tuples that not satisfy the constant predicates
                for attr, v in constants_in_X[0]:  # constant predicates
                    tuples_check = tuples_repaired.loc[reserved_indices]
                    reserved_indices = tuples_check.loc[tuples_check[attr].astype(str) == v].index.values
                    if reserved_indices.shape[0] == 0:
                        break
                if reserved_indices.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                    continue
                # (2) then construct dict by non-constant predicates
                for value, df in tuples_repaired.loc[reserved_indices].groupby(key_attributes_non_constant):
                    value2tid_tuples[value] = df  # df.index.values
                if len(value2tid_tuples) == 0:  # there's no tuples satisfy the non-constant predicates in X of current ree; we should go for the next ree
                    continue
                # 3. get value of GT
                # (1) first filter tuples that not satisfy the constant predicates
                reserved_indices_GT = GT.index
                for attr, v in constants_in_X[1]:  # constant predicates
                    tuples_check_constants = GT.loc[reserved_indices_GT]
                    reserved_indices_GT = tuples_check_constants.loc[tuples_check_constants[attr].astype(str) == v].index.values
                    if reserved_indices_GT.shape[0] == 0:
                        break
                if reserved_indices_GT.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
                    continue
                # (2) then construct dict by non-constant predicates
                for value, df in GT.loc[reserved_indices_GT].groupby(key_attributes_non_constant):
                    value2tid_GT[value] = df  # df.index.values
                if len(value2tid_GT) == 0:  # there's no tuples satisfy the non-constant predicates in X of current ree; we should go for the next ree
                    continue
                # 4. check Y
                violate_indices = []
                # (1) check constant Y
                if rhs.get_type() == "constant":
                    for key in value2tid_tuples.keys():
                        if key not in value2tid_GT.keys():
                            continue
                        df = value2tid_tuples[key]
                        null_indices = df.loc[df[attr_RHS].isnull()].index.values
                        violateY_indices = df.loc[df[attr_RHS].astype(str) != rhs.get_constant()].index.values
                        violateY_indices = np.array(list(set(violateY_indices).difference(set(null_indices))))
                        if violateY_indices.shape[0] > 0:
                            violate_indices.append(violateY_indices)
                    for indices in violate_indices:
                        tuples_repaired.loc[indices, attr_RHS] = None
                        tuple_status[0].loc[indices, attr_RHS] = 2  # repair
                        tuple_status[1].loc[indices, attr_RHS] = 2  # repair
                # (2) check non-constant Y
                else:
                    for key in value2tid_tuples.keys():
                        if key not in value2tid_GT.keys():
                            continue
                        check_attr = attr_RHS
                        check_series = value2tid_tuples[key][check_attr]
                        for index_t_ in check_series.index:
                            value_in_tuples = value2tid_tuples[key].loc[index_t_][check_attr]
                            if pd.isnull(value_in_tuples) is True:  # None can not be regarded as violate the rules
                                continue
                            if value2tid_GT[key].loc[value2tid_GT[key][check_attr].astype(str) != str(value_in_tuples)].index.values.shape[0] != 0:  # as long as there exists one violation, then the tuple has conflict
                                violate_indices.append(index_t_)
                    if len(violate_indices) > 0:
                        tuples_repaired.loc[np.array(violate_indices), attr_RHS] = None
                        tuple_status[0].loc[np.array(violate_indices), attr_RHS] = 2  # repair
                        tuple_status[1].loc[np.array(violate_indices), attr_RHS] = 2  # repair

    if useMc:
        for attr_to_check in reserved_attrs:
            remain_indices = remain_attr_indices[attr_to_check]
            isnull_indices = tuples_repaired.loc[tuples_repaired[attr_to_check].isnull()].index.values
            remain_indices = np.array(list(set(remain_indices).difference(set(isnull_indices))))  # the indices of tuples with current non-null single attribute

            if remain_indices.shape[0] == 0:
                continue

            basic_df.loc[remain_indices, attr_to_check] = tuples_repaired.loc[remain_indices, attr_to_check]

            scores = Mc_model.predictMcScore_new(basic_df.loc[remain_indices])
            scores = [float(i) for i in scores]
            violate_tuple_index = np.argwhere(np.array(scores) >= Mc_conf)
            if violate_tuple_index.shape[0] > 0:
                violate_tuple_index = np.array([remain_indices[idx[0]] for idx in violate_tuple_index])
                basic_df.loc[violate_tuple_index, attr_to_check] = None
                tuples_repaired.loc[violate_tuple_index, attr_to_check] = None
                tuple_status[0].loc[violate_tuple_index, attr_to_check] = 2  # repair
                tuple_status[1].loc[violate_tuple_index, attr_to_check] = 2  # repair

    return tuples_repaired, tuple_status


# for Imp3C and Holoclean
def repair_HyperCube(rees, GT, tuples, exist_conflict, output_path, useREE, tuple_status, whether_output_results):
    repair_indices = np.where(exist_conflict == -1)[0]
    no_conflict_indices = np.where(exist_conflict == 0)[0]

    if repair_indices.shape[0] == 0:
        return None, None

    all_attrs = tuples.columns.values.tolist()
    new_tuples = [None, None]
    for tid in [0, 1]:
        new_tuples[tid] = np.full((tuples.shape[0], len(all_attrs)), None)
        new_tuples[tid] = pd.DataFrame(new_tuples[tid], columns=all_attrs)

    if no_conflict_indices.shape[0] != 0:
        for tid in [0, 1]:
            new_tuples[tid].loc[no_conflict_indices, :] = tuples.loc[no_conflict_indices, :]

    tuples_repaired, tuple_status = repair_tuples(tuples, repair_indices, useREE, False, rees, GT, None, None, tuple_status)
    new_tuples[0].loc[repair_indices] = tuples_repaired
    new_tuples[1].loc[repair_indices] = tuples_repaired

    # save repaired results to files
    if whether_output_results is True:
        new_tuples[0].to_csv(output_path + "tuples_splitting_t0.csv", index=False)
        new_tuples[1].to_csv(output_path + "tuples_splitting_t1.csv", index=False)

    return new_tuples, tuple_status
