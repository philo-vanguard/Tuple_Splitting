import numpy as np
import pandas as pd
import operator
import copy

operators = {'=': operator.eq, '<>': operator.ne, '>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le}


# load dict from the begging
def predictMcScore(t_barA_B, predictor=None, wikidata_embed=None, wikidata_dic=None, item_dict=None, item_dict_reverse=None):
    def wiki_embed_load_fix_length(List):
        length = len(List)
        embed = np.zeros((length,200))
        for l in range(length):
            if pd.isnull(List[l]):
                embed[l] = np.zeros((200))
            else:
                embed[l] = wikidata_embed[wikidata_dic[int(List[l])]]
        return embed

    def AssignEmbed(row):
        seq = row[:11].values
        return wiki_embed_load_fix_length(seq).reshape((2200))

    if predictor is None:
        # for test.
        if len(t_barA_B.shape) == 1:
            scores = [1.0]
        else:
            scores = [1.0] * t_barA_B.shape[0]
        return scores

    ## convert to uniform columns
    col_seq = ['full name', 'given name', 'family name', 'place of birth', 'place of death', 'gender', 'country',
               'achieve', 'occupation', 'educated at', 'member of']

    if len(t_barA_B.shape) == 1:
        test_all = t_barA_B.to_frame().transpose()
    else:
        test_all = copy.deepcopy(t_barA_B)

    if "id" in test_all.columns:
        test_all.drop(columns=["id"], inplace=True)

    old_columns = test_all.columns.tolist()
    for index, value in enumerate(old_columns):
        old_columns[index] = value.replace("_", " ")
    test_all.columns = old_columns

    test_all = test_all.reindex(columns=col_seq, fill_value=np.nan)
    for c in col_seq:
        test_all[c] = test_all[c].map(item_dict_reverse[c])
    AutoML = pd.DataFrame(columns=np.arange(0, 2200, 1))

    AutoML = test_all.apply(AssignEmbed, axis=1, result_type='expand')

    predict_result = predictor.predict_proba(AutoML)

    scores = predict_result[1].values

    return scores


def satisfyPredicate(predicate, t_, t2, Mc_model=None):
    operator = predicate.get_operator()
    if predicate.get_type() == "constant":
        attr1 = predicate.get_attr1()
        constant = predicate.get_constant()
        # if predicate.get_operator() == "=" and t_[attr1] == constant and constant != "nan":
        # if operators[operator](str(t_[attr1]), constant) and not pd.isnull(constant): # normal version
        if (operators[operator](str(t_[attr1]), constant) and not pd.isnull(constant)) or str(t_[attr1]) == "": # relaxed for partial satisfication by ignore null values
            return True

    elif predicate.get_type() == "non-constant":
        attr1 = predicate.get_attr1()
        attr2 = predicate.get_attr2()
        # if predicate.get_operator() == "=" and t_[attr1] == t2[attr2] and t_[attr1] != "nan":
        # if operators[operator](str(t_[attr1]), str(t2[attr2])) and not pd.isnull(t_[attr1]): # normal version
        if (operators[operator](str(t_[attr1]), str(t2[attr2])) and not pd.isnull(t_[attr1])) or str(t_[attr1]) == "" or str(t2[attr2]) == "": # relaxed for partial satisfication by ignore null values
            return True

    elif predicate.get_type() == "M_c":
        bar_A = predicate.get_attr1()
        B = predicate.get_attr2()
        confidence = predicate.get_confidence()
        if Mc_model is not None:
            score = Mc_model.predictMcScore_new(t_[bar_A + [B]])[0]
        else:
            score = predictMcScore(t_barA_B=t_[bar_A + [B]])[0]
        # if operator == ">" and score > confidence:
        if operators[operator](score, confidence):
            return True

    return False


# True  - 'satisfy X but not Y'
# False - 'not satisfy X' or 'satisfy X and Y'
def violateSingleRule(rule, t_, Mc_model=None):
    violate = False
    splitting_attr = ""

    if rule.get_type() == "logic":
        satisfyX = True
        for predicate in rule.get_currents():
            if satisfyPredicate(predicate, t_, None) is False:
                satisfyX = False
                break
        if satisfyX is True and satisfyPredicate(rule.get_RHS(), t_, None) is False:
            violate = True
            splitting_attr = rule.get_RHS().get_attr1()  # RHS attribute

    elif rule.get_type() == "M_c":
        if satisfyPredicate(rule.get_currents(), t_, None, Mc_model) is True:
            violate = True
            splitting_attr = rule.get_currents().get_attr2()  # B attribute

    return violate, splitting_attr


# True  - 'satisfy X but not Y'
# False - 'not satisfy X' or 'satisfy X and Y'
def violateBiVariableREE(ree, t_, t2):
    violate = False
    splitting_attr = ""

    satisfyX = True
    for predicate in ree.get_currents():
        if satisfyPredicate(predicate, t_, t2) is False:
            satisfyX = False
            break
    if satisfyX is True and satisfyPredicate(ree.get_RHS(), t_, t2) is False:
        violate = True
        splitting_attr = ree.get_RHS().get_attr1()

    return violate, splitting_attr


def filter_useful_rees(rees, assign_attribute):
    useful_single_rees, useful_bi_rees = [], []
    for ree in rees:
        e_attribute = ree.get_RHS().get_attr1()
        if e_attribute == assign_attribute:
            if ree.get_tuple_variable_cnt() == 1:
                useful_single_rees.append(ree)
            else:
                useful_bi_rees.append(ree)
    return useful_single_rees, useful_bi_rees


# ============================= below is for checking whether one tuple with the entire Groundtruth are satisfied for a REE rule =============================

# True: t_ satisfy the constant predicate; or t_ satisfy the non-constant predicate with at least a tuple in GT
# False: t_ cannot satisfy the predicate with any tuple in GT
# return indices: the reserved indices of tuple in GT that satisfy the predicate with combination with t_
def satisfyPredicate_ForTuples(predicate, t_, GT, tid_t_):
    operator = predicate.get_operator()
    if predicate.get_type() == "constant":
        index1 = predicate.get_index1()
        attr1 = predicate.get_attr1()
        constant = predicate.get_constant()
        if tid_t_ == index1:
            if operators[operator](t_[attr1], constant) and not pd.isnull(constant):
                return True, GT.index.values
            else:
                return False, None
        else:
            reserved_indices_tuples = GT.loc[operators[operator](GT[attr1], constant)].index.values
            return reserved_indices_tuples.shape[0] != 0, reserved_indices_tuples

    elif predicate.get_type() == "non-constant":
        attr1 = predicate.get_attr1()
        attr2 = predicate.get_attr2()
        if tid_t_ == 0:  # t_ is t0
            reserved_indices_tuples = GT.loc[operators[operator](t_[attr1], GT[attr2])].index.values
            return reserved_indices_tuples.shape[0] != 0, reserved_indices_tuples
        else:            # t_ is t1
            reserved_indices_tuples = GT.loc[operators[operator](GT[attr1], t_[attr2])].index.values
            return reserved_indices_tuples.shape[0] != 0, reserved_indices_tuples


# checking whether any of the tuple in 'tuples' violate the rule
def violateSingleRule_ForTuples(rule, tuples):
    violate = False
    splitting_attr = ""
    violate_rule_indices = None

    if rule.get_type() == "logic":
        reserved_indices = tuples.index.values
        for predicate in rule.get_currents():
            attr = predicate.get_attr1()
            operator = predicate.get_operator()
            constant = predicate.get_constant()
            reserved_tuples_check = tuples.loc[reserved_indices]
            reserved_indices = reserved_tuples_check.loc[operators[operator](reserved_tuples_check[attr].astype(str), constant)].index.values
            if reserved_indices.shape[0] == 0:
                break

        if reserved_indices.shape[0] != 0:  # there exist tuples that satisfy the X of rule
            # reserved_indices_satisfyX = [i for i in reserved_indices]
            rhs = rule.get_RHS()
            attr = rhs.get_attr1()
            operator = rhs.get_operator()
            constant = rhs.get_constant()
            reserved_tuples_check = tuples.loc[reserved_indices]
            not_null_indices = reserved_tuples_check.loc[pd.notnull(reserved_tuples_check[attr])].index.values
            satisfyY_indices = reserved_tuples_check.loc[operators[operator](reserved_tuples_check[attr].astype(str), constant)].index.values  # the indices of tuples that both satisfy X and Y
            # violate_rule_indices = [i for i in not_null_indices if i not in satisfyY_indices]
            violate_rule_indices = list(set(not_null_indices).difference(set(satisfyY_indices)))  # list(set(a).difference(set(b)))  # a contains but b does not hold
            if len(violate_rule_indices) != 0:
                violate = True
                splitting_attr = attr

    return violate, splitting_attr, violate_rule_indices


# True  - 'satisfy X but not Y' for any tuple in GT
# False - 'not satisfy X' or 'satisfy X and Y'
def violateBiVariableREE_ForTuples(ree, t_, GT):
    index_t_ = [False, False]
    rhs = ree.get_RHS()
    if rhs.get_type() == "constant":
        index_t_[rhs.get_index1()] = True
    elif rhs.get_type() == "non-constant":
        index_t_ = [True, True]

    violate = False
    splitting_attr = ""

    for tid_t_ in [0, 1]:
        if index_t_[tid_t_] is False:
            continue
        reserved_indices = GT.index.values
        satisfyX = True
        for predicate in ree.get_currents():
            attr1 = predicate.get_attr1() # the following three lines are for relaxing the valuations of REEs
            if str(t_[attr1]) == "":
                continue
            satisfy, reserved_indices = satisfyPredicate_ForTuples(predicate, t_, GT.loc[reserved_indices], tid_t_)
            if satisfy is False:
                satisfyX = False
                break
        if satisfyX is True:
            num_tupls_satisfy_X = reserved_indices.shape[0]
            satisfy, reserved_indices = satisfyPredicate_ForTuples(rhs, t_, GT.loc[reserved_indices], tid_t_)
            if satisfy is False or num_tupls_satisfy_X != reserved_indices.shape[0]:  # If there's any one tuple together with t_ violate RHS
                violate = True  # satisfy X but not Y
                splitting_attr = rhs.get_attr1()

    return violate, splitting_attr


# ============================= below is for checking whether a tuple t_ has already violate the constant predicates in bi-variable rules, so that there's no need to check other tuples in groundtruth together with t =============================
# check whether tuple t satisfy the constant predicates in X; If not, then there's no need to check other tuples in groundtruth together with t
def satisfyConstantPredicatesInX(ree, t):
    constant_predicates_in_X = [[], []]  # t0, t1 related
    for predicate in ree.get_currents():
        if predicate.get_type() == "constant":
            constant_predicates_in_X[predicate.get_index1()].append(predicate)

    satify_cpredicates = [True, True]
    for tid in [0, 1]:
        for predicate in constant_predicates_in_X[tid]:
            if satisfyPredicate(predicate, t, None) is False:
                satify_cpredicates[tid] = False
                break

    return satify_cpredicates


# check whether some tuples t satisfy the constant predicates in X; If not, then there's no need to check other tuples in groundtruth together with t
# this function is expensive and useless, because one tuple maybe violate one ree and we know it should be split and we dont need to check the other left rees; but if we calculate the info by this function, we check all rees in advance and some of them are no need to check
def satisfyConstantPredicatesInX_forTuples(ree, tuples):
    constant_predicates_in_X = [[], []]  # t0, t1 related
    for predicate in ree.get_currents():
        if predicate.get_type() == "constant":
            constant_predicates_in_X[predicate.get_index1()].append(predicate)

    satify_cpredicates = [[True, True]] * tuples.shape[0]
    for index, row in tuples.iterrows():
        for tid in [0, 1]:
            for predicate in constant_predicates_in_X[tid]:
                if satisfyPredicate(predicate, row, None) is False:
                    satify_cpredicates[index][tid] = False
                    break

    return satify_cpredicates


def filter_tuples_satisfy_at_least_one_rule(tuples, rees, GT):
    single_rees, bi_rees = [], []
    for ree in rees:
        if ree.get_tuple_variable_cnt() == 1:
            single_rees.append(ree)
        else:
            bi_rees.append(ree)

    satisfy_one_rule = pd.Series([False] * tuples.shape[0])
    satisfy_one_rule.index = tuples.index

    for ree in single_rees:
        reserved_indices = satisfy_one_rule.loc[satisfy_one_rule == False].index.values
        if reserved_indices.shape[0] == 0:
            break
        for predicate in ree.get_currents():
            attr = predicate.get_attr1()
            constant = predicate.get_constant()
            reserved_tuples_check = tuples.loc[reserved_indices]
            reserved_indices = reserved_tuples_check.loc[reserved_tuples_check[attr].astype(str) == constant].index.values
            if reserved_indices.shape[0] == 0:
                break
        if reserved_indices.shape[0] != 0:
            rhs = ree.get_RHS()
            attr = rhs.get_attr1()
            constant = rhs.get_constant()
            reserved_tuples_check = tuples.loc[reserved_indices]
            reserved_indices = reserved_tuples_check.loc[reserved_tuples_check[attr].astype(str) == constant].index.values
            satisfy_one_rule.loc[reserved_indices] = True

    for ree in bi_rees:
        rhs = ree.get_RHS()
        if rhs.get_type() == "constant" and rhs.get_index1() == 1:
            continue

        reserved_indices = satisfy_one_rule.loc[satisfy_one_rule == False].index.values
        if reserved_indices.shape[0] == 0:
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
                attr = rhs.get_attr1()
                constant = rhs.get_constant()
                reserved_tuples_check = tuples.loc[indices]
                reserved_indices = reserved_tuples_check.loc[reserved_tuples_check[attr].astype(str) == constant].index.values
                satisfy_one_rule.loc[reserved_indices] = True

        # apply non-constant Y
        else:
            for key in value2tid_tuples.keys():
                if key not in value2tid_GT.keys():
                    continue
                attr_1 = rhs.get_attr1()
                attr_2 = rhs.get_attr2()
                series_values_tuples = value2tid_tuples[key][attr_1]
                series_values_GT = value2tid_GT[key][attr_2]
                indices = value2tid_tuples[key].index.values
                for index in indices:
                    equal = (series_values_GT.astype(str) == str(series_values_tuples.loc[index]))
                    contain_true_indices = equal.loc[equal == True].index.values
                    if contain_true_indices.shape[0] != 0:
                        satisfy_one_rule.loc[index] = True

    addable_indices = satisfy_one_rule.loc[satisfy_one_rule == True].index.values
    add_tuples = tuples.loc[addable_indices]

    return add_tuples


def filter_tuples_add_to_Gamma(tuples, exist_conflict, num_attributes, GT, rees):
    no_split_indices = np.where(exist_conflict == 0)[0]
    positive_indices = tuples.loc[tuples["label"] == 0].index.values
    add_indices = np.array([i for i in no_split_indices if i in positive_indices])

    if add_indices.shape[0] == 0:
        return None

    add_tuples_with_label = copy.deepcopy(tuples.loc[add_indices])
    add_tuples = add_tuples_with_label.iloc[:, :num_attributes]
    add_tuples = pd.concat([add_tuples, add_tuples_with_label["id"]], axis=1)

    # check which id the attr values belong to
    ids_in_GT = list(set(GT["id"].astype(str).values))
    belong_indices = pd.Series([None] * add_tuples.shape[0])
    belong_indices.index = add_indices
    all_attrs = [i for i in add_tuples.columns if i != "id"]
    for attr in all_attrs:
        check_indices = belong_indices.loc[belong_indices.isnull()].index.values
        if check_indices.shape[0] == 0:
            break
        check_tuples = tuples.loc[check_indices]
        belong_t1_indices = check_tuples.loc[check_tuples["label_" + attr].astype(str).str.contains("-1")].index.values
        belong_t0_indices = np.array([i for i in check_indices if i not in belong_t1_indices])
        belong_indices.loc[belong_t0_indices] = 0
        belong_indices.loc[belong_t1_indices] = 1

    # assign the values
    assign_t0_indices = belong_indices.loc[belong_indices == 0].index.values
    assign_t1_indices = belong_indices.loc[belong_indices == 1].index.values
    for attr in (["id"] + all_attrs):
        if add_tuples.shape[0] == 0:
            return None
        contains_vertical_virgule_indices = add_tuples.loc[add_tuples[attr].astype(str).str.contains("\|\|")].index.values
        contains_sharp_indices = add_tuples.loc[add_tuples[attr].astype(str).str.contains("##")].index.values

        # assign values split by ||
        assign_t0_vertical_virgule_indices = np.array([i for i in assign_t0_indices if i in contains_vertical_virgule_indices])
        assign_t1_vertical_virgule_indices = np.array([i for i in assign_t1_indices if i in contains_vertical_virgule_indices])
        add_tuples.loc[assign_t0_vertical_virgule_indices, attr] = add_tuples.loc[assign_t0_vertical_virgule_indices, attr].map(lambda x: x.split("||")[0]).replace("", np.nan)
        add_tuples.loc[assign_t1_vertical_virgule_indices, attr] = add_tuples.loc[assign_t1_vertical_virgule_indices, attr].map(lambda x: x.split("||")[1]).replace("", np.nan)

        # assign values split by ##
        assign_t0_sharp_indices = np.array([i for i in assign_t0_indices if i in contains_sharp_indices])
        assign_t1_sharp_indices = np.array([i for i in assign_t1_indices if i in contains_sharp_indices])
        add_tuples.loc[assign_t0_sharp_indices, attr] = add_tuples.loc[assign_t0_sharp_indices, attr].map(lambda x: x.split("##")[0]).replace("", np.nan)
        add_tuples.loc[assign_t1_sharp_indices, attr] = add_tuples.loc[assign_t1_sharp_indices, attr].map(lambda x: x.split("##")[1]).replace("", np.nan)

        addable_indices = add_tuples[~add_tuples["id"].astype(str).isin(ids_in_GT)].index.values
        add_tuples = add_tuples.loc[addable_indices]

    # filter the tuples with id exists in Gamma already
    ids_in_GT = list(set(GT["id"].astype(str).values))
    addable_indices = add_tuples[~add_tuples["id"].astype(str).isin(ids_in_GT)].index.values
    add_tuples = add_tuples.loc[addable_indices]

    # filter the tuples that at least satisfy one rule
    add_tuples = filter_tuples_satisfy_at_least_one_rule(add_tuples, rees, GT)

    if add_tuples.shape[0] == 0:
        return None
    else:
        return add_tuples


def removeSharp(tuples):
    for attr in tuples.columns.values:
        contain_sharp_indices = tuples.loc[tuples[attr].astype(str).str.contains("##", na=False)].index.values
        if contain_sharp_indices.shape[0] == 0:
            continue
        tuples.loc[contain_sharp_indices, attr] = tuples.loc[contain_sharp_indices, attr].map(lambda x: x.split("##")[1]).replace("", np.nan)
    return tuples


def obtain_groundtruth_splitting_results(merged_tuples, number_attributes, output_path, whether_output_results):
    labels = merged_tuples["label"]
    split_indices = labels.loc[labels == 1].index.values
    repair_indices = labels.loc[labels == -1].index.values
    conflict_indices = np.array(list(set(split_indices).union(set(repair_indices))))
    not_conflict_indices = labels.loc[labels == 0].index.values

    tuples = merged_tuples.iloc[:, 0:number_attributes]
    tuples = pd.concat([tuples, merged_tuples["id"]], axis=1)

    all_attrs = tuples.columns.values.tolist()
    new_tuples = [None, None]
    for tid in [0, 1]:
        new_tuples[tid] = np.full((tuples.shape[0], len(all_attrs)), None)
        new_tuples[tid] = pd.DataFrame(new_tuples[tid], columns=all_attrs)

    # (1) assign values for tuples that should be split and repaired
    tuples_check = tuples.loc[conflict_indices]
    for attr in all_attrs:
        # deal with ||
        contains_vertical_virgule_indices = tuples_check.loc[tuples_check[attr].astype(str).str.contains("\|\|", na=False)].index.values
        new_tuples[0].loc[contains_vertical_virgule_indices, attr] = tuples.loc[contains_vertical_virgule_indices, attr].map(lambda x: x.split("||")[0]).replace("", np.nan)
        new_tuples[1].loc[contains_vertical_virgule_indices, attr] = tuples.loc[contains_vertical_virgule_indices, attr].map(lambda x: x.split("||")[1]).replace("", np.nan)

        # deal with ##
        contains_sharp_indices = tuples_check.loc[tuples_check[attr].astype(str).str.contains("##", na=False)].index.values
        new_tuples[0].loc[contains_sharp_indices, attr] = tuples.loc[contains_sharp_indices, attr].map(lambda x: x.split("##")[0]).replace("", np.nan)
        new_tuples[1].loc[contains_sharp_indices, attr] = tuples.loc[contains_sharp_indices, attr].map(lambda x: x.split("##")[1]).replace("", np.nan)

        # deal with single value
        not_null_indices = tuples_check.loc[pd.notnull(tuples_check[attr])].index.values
        # single_value_indices = np.array([i for i in not_null_indices if (i not in contains_vertical_virgule_indices and i not in contains_sharp_indices)])
        contain_marker_indices = list(set(contains_vertical_virgule_indices).union(set(contains_sharp_indices)))
        single_value_indices = np.array(list(set(not_null_indices).difference(set(contain_marker_indices))))
        if single_value_indices.shape[0] == 0:
            continue
        label_attr = merged_tuples.loc[single_value_indices, "label_" + attr]
        assign_t0_indices = label_attr.loc[label_attr == 1].index.values
        assign_t1_indices = label_attr.loc[label_attr == -1].index.values
        assign_both_indices = label_attr.loc[label_attr == 0].index.values
        to_be_repaired_indices = label_attr.loc[label_attr == -2].index.values
        new_tuples[0].loc[assign_t0_indices, attr] = tuples.loc[assign_t0_indices, attr]
        new_tuples[0].loc[assign_both_indices, attr] = tuples.loc[assign_both_indices, attr]
        new_tuples[0].loc[to_be_repaired_indices, attr] = None
        new_tuples[1].loc[assign_t1_indices, attr] = tuples.loc[assign_t1_indices, attr]
        new_tuples[1].loc[assign_both_indices, attr] = tuples.loc[assign_both_indices, attr]
        new_tuples[1].loc[to_be_repaired_indices, attr] = None

    # repaired tuples only keep the latest one, i.e., t1 instead of t0
    new_tuples[0].loc[repair_indices] = new_tuples[1].loc[repair_indices]

    # (2) assign values for tuples that no need be split
    # a. check which id the attr values belong to
    belong_indices = pd.Series([None] * not_conflict_indices.shape[0])
    belong_indices.index = not_conflict_indices
    all_attrs.remove("id")
    for attr in all_attrs:
        check_indices = belong_indices.loc[belong_indices.isnull()].index.values
        if check_indices.shape[0] == 0:
            break
        label_attr = merged_tuples.loc[check_indices, "label_" + attr]
        belong_t0_indices = label_attr.loc[label_attr == 1].index.values
        belong_t1_indices = label_attr.loc[label_attr == -1].index.values
        belong_indices.loc[belong_t0_indices] = 0
        belong_indices.loc[belong_t1_indices] = 1
    # for tuples that merged by the same tuples, so that the labels of all attrs are all be 0
    remaining_indices = belong_indices.loc[belong_indices.isnull()].index.values
    belong_indices.loc[remaining_indices] = 0

    # b. assign values
    assign_t0_indices = belong_indices.loc[belong_indices == 0].index.values
    assign_t1_indices = belong_indices.loc[belong_indices == 1].index.values
    all_attrs.append("id")
    for tid in [0, 1]:
        for attr in all_attrs:
            tuples_check_t0 = tuples.loc[assign_t0_indices]
            tuples_check_t1 = tuples.loc[assign_t1_indices]

            # deal with ||
            contains_vertical_virgule_indices_t0 = tuples_check_t0.loc[tuples_check_t0[attr].astype(str).str.contains("\|\|", na=False)].index.values
            contains_vertical_virgule_indices_t1 = tuples_check_t1.loc[tuples_check_t1[attr].astype(str).str.contains("\|\|", na=False)].index.values
            new_tuples[tid].loc[contains_vertical_virgule_indices_t0, attr] = tuples.loc[contains_vertical_virgule_indices_t0, attr].map(lambda x: x.split("||")[0]).replace("", np.nan)
            new_tuples[tid].loc[contains_vertical_virgule_indices_t1, attr] = tuples.loc[contains_vertical_virgule_indices_t1, attr].map(lambda x: x.split("||")[1]).replace("", np.nan)

            # deal with ##
            contains_sharp_indices_t0 = tuples_check_t0.loc[tuples_check_t0[attr].astype(str).str.contains("##", na=False)].index.values
            contains_sharp_indices_t1 = tuples_check_t1.loc[tuples_check_t1[attr].astype(str).str.contains("##", na=False)].index.values
            new_tuples[tid].loc[contains_sharp_indices_t0, attr] = tuples.loc[contains_sharp_indices_t0, attr].map(lambda x: x.split("##")[0]).replace("", np.nan)
            new_tuples[tid].loc[contains_sharp_indices_t1, attr] = tuples.loc[contains_sharp_indices_t1, attr].map(lambda x: x.split("##")[1]).replace("", np.nan)

            # deal with single value
            not_null_indices_t0 = tuples_check_t0.loc[pd.notnull(tuples_check_t0[attr])].index.values
            not_null_indices_t1 = tuples_check_t1.loc[pd.notnull(tuples_check_t1[attr])].index.values

            # single_value_indices_t0 = np.array([i for i in not_null_indices_t0 if (i not in contains_vertical_virgule_indices_t0 and i not in contains_sharp_indices_t0)])
            contain_marker_indices_t0 = list(set(contains_vertical_virgule_indices_t0).union(set(contains_sharp_indices_t0)))
            single_value_indices_t0 = np.array(list(set(not_null_indices_t0).difference(set(contain_marker_indices_t0))))

            # single_value_indices_t1 = np.array([i for i in not_null_indices_t1 if (i not in contains_vertical_virgule_indices_t1 and i not in contains_sharp_indices_t1)])
            contain_marker_indices_t1 = list(set(contains_vertical_virgule_indices_t1).union(set(contains_sharp_indices_t1)))
            single_value_indices_t1 = np.array(list(set(not_null_indices_t1).difference(set(contain_marker_indices_t1))))

            new_tuples[tid].loc[single_value_indices_t0, attr] = tuples.loc[single_value_indices_t0, attr]
            new_tuples[tid].loc[single_value_indices_t1, attr] = tuples.loc[single_value_indices_t1, attr]

    if whether_output_results is True:
        new_tuples[0].to_csv(output_path + "tuples_splitting_groundtruth_t0.csv", index=False)
        new_tuples[1].to_csv(output_path + "tuples_splitting_groundtruth_t1.csv", index=False)

    return new_tuples
