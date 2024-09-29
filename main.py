import copy
import sys

import numpy as np

from TupleSplitting.splitTuples import *
from Imputation.imputation import *
import argparse
import logging
from Evaluation.evaluation import *
import os
from TupleSplitting.M_c import CorrelationModel
from Imputation.M_d import ImputationModel
from Imputation.her import HER
from pandarallel import pandarallel
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 1000)
# pd.option.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)


def str_to_bool(str):
    return True if str.lower() == 'true' else False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Given knowledge graph KG, groundtruth GT, REEs Sigma, Md rules Sigma_Md, and splitting tuples, impute the missing values of these tuples")
    parser.add_argument('-groundtruth', '--groundtruth', type=str, default="data/test_data/groundtruth.csv")
    parser.add_argument('-ree_path', '--ree_path', type=str, default="data/test_data/logic_rules.txt")
    parser.add_argument('-Mc_path', '--Mc_path', type=str, default="data/test_data/Mc_rules.txt")
    parser.add_argument('-tuples_path', "--tuples_path", type=str, default="data/test_data/tuples_check.csv")
    parser.add_argument('-data_name', '--data_name', type=str, default="persons")
    parser.add_argument('-main_attribute', "--main_attribute", type=str, default="")
    parser.add_argument('-output_exist_conflict_dir', '--output_exist_conflict_dir', type=str, default="data/output/")
    parser.add_argument('-output_tuples_splitting_dir', '--output_tuples_splitting_dir', type=str, default="data/output/")
    parser.add_argument("-output_tuples_imputation_dir", '--output_tuples_imputation_dir', type=str, default="data/output/")
    parser.add_argument("-output_updated_groundtrutrh", '--output_updated_groundtrutrh', type=str, default="data/output/groundtruth_updated.csv")
    parser.add_argument("-output_tmp_files_for_Md_dir", '--output_tmp_files_for_Md_dir', type=str, default="data/output/")

    parser.add_argument('-useREE', '--useREE', type=str_to_bool, default=True)  # whether use logic rules for splitting and imputation
    parser.add_argument('-useMc', '--useMc', type=str_to_bool, default=True)  # whether use Mc rules for splitting and imputation
    parser.add_argument('-useKG', '--useKG', type=str_to_bool, default=True)  # whether use knowledge graph for imputation
    parser.add_argument('-useMd', '--useMd', type=str_to_bool, default=True)  # whether use Md rules for imputation
    parser.add_argument('-useBFchase', '--useBFchase', type=str_to_bool, default=False)  # whether use brute-force chase for imputation

    parser.add_argument('-varyGT', '--varyGT', type=str_to_bool, default=False)  # vary Groundtruth - vary Gamma
    parser.add_argument('-varyKG', '--varyKG', type=str_to_bool, default=False)  # vary Knowledge Graph - vary G
    parser.add_argument('-varyREE', '--varyREE', type=str_to_bool, default=False)  # vary REE size - vary Sigma
    parser.add_argument('-varyTuples', '--varyTuples', type=str_to_bool, default=False)  # vary tuples - vary D_T
    parser.add_argument('-varyMc', '--varyMc', type=str_to_bool, default=False)  # vary training data size of Mc - vary M_c
    parser.add_argument('-varyMd', '--varyMd', type=str_to_bool, default=False)  # vary training data size of Md - vary M_d
    parser.add_argument('-vary_ratio', '--vary_ratio', type=float, default=1.0)  # the ratio for varying stuff
    parser.add_argument('-default_ratio', '--default_ratio', type=float, default=1.0)  # the ratio for the rest that no need to vary size, except KG and the training ratio of Mc and Md
    parser.add_argument('-default_KG_ratio', '--default_KG_ratio', type=float, default=1.0)  # the ratio of KG
    parser.add_argument('-default_training_ratio', '--default_training_ratio', type=float, default=0.2)  # the training ratio of Mc and Md model

    parser.add_argument('-useImp3C', '--useImp3C', type=str_to_bool, default=False)  # baseline - Imp3C
    parser.add_argument('-useHoloClean', '--useHoloClean', type=str_to_bool, default=False)  # baseline - HoloClean

    parser.add_argument('-cuda', '--cuda', type=int, default=3)
    parser.add_argument('-if_parallel', '--if_parallel', type=str_to_bool, default=False)
    parser.add_argument('-max_chase_round', '--max_chase_round', type=int, default=100)

    parser.add_argument('-Mc_type', '--Mc_type', type=str, default="graph")  # type of Mc model: graph, bert, ensemble
    parser.add_argument('-Mc_conf', '--Mc_conf', type=float, default=0.5)  # the confidence of Mc model
    parser.add_argument('-evaluate_separately', '--evaluate_separately', type=str_to_bool, default=True)  # whether separately evaluate DecideTS, Splitting, and Imputation. If so, we should use the groundtruth "exist_conflict" for Splitting phase; and use the groundtruth "splitting_tuples" for Imputation phase.
    parser.add_argument('-impute_multi_values', '--impute_multi_values', type=str_to_bool, default=False)  # whether impute all empty values of one attribute in one round when using HER
    parser.add_argument('-use_Mc_rules', '--use_Mc_rules', type=str_to_bool, default=True)  # whether use Mc rules when DecideTS; If not, then use Mc model directly
    parser.add_argument('-run_DecideTS', '--run_DecideTS', type=str_to_bool, default=True)  # whether run DecideTS part
    parser.add_argument('-run_Splitting', '--run_Splitting', type=str_to_bool, default=True)  # whether run Splitting part
    parser.add_argument('-run_Imputation', '--run_Imputation', type=str_to_bool, default=True)  # whether run Imputation part
    parser.add_argument('-evaluateDS_syn', '--evaluateDS_syn', type=str_to_bool, default=False)  # True: evaluate synthetic tuples_check, use "and"; False: evaluate ER false positive, use "or";

    parser.add_argument('-useDittoRules', '--useDittoRules', type=str_to_bool, default=True)  # whether use ditto rules in DecideTS
    parser.add_argument('-thr_ditto_rule', '--thr_ditto_rule', type=float, default=0.5)  # the threshold of confidence in ditto rules for DecideTS

    parser.add_argument('-exist_AA', '--exist_AA', type=str_to_bool, default=True)  # True: compute F1 for AA and MI separately; False: compute F1 for AA+MI.
    parser.add_argument('-output_results', '--output_results', type=str_to_bool, default=False)  # whether save the intermediate and final results into files
    parser.add_argument('-isSET', '--isSET', type=str_to_bool, default=False)  # whether the current method running is SET

    parser.add_argument('-count_violations', '--count_violations', type=str_to_bool, default=False)  # whether output the number of #violation
    parser.add_argument('-track_tid', '--track_tid', type=int, default=None)  # the id of tuple been tracked, to show the process of DS, AA, MI for it

    random.seed(7654321)
    np.random.seed(7654321)

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))

    useREE, useKG, useMd, useMc, useBFchase = arg_dict["useREE"], arg_dict["useKG"], arg_dict["useMd"], arg_dict["useMc"], arg_dict["useBFchase"]
    useImp3C, useHoloClean = arg_dict["useImp3C"], arg_dict["useHoloClean"]
    print("useREE: ", useREE)
    print("useMc: ", useMc)
    print("useKG: ", useKG)
    print("useMd: ", useMd)
    print("useBFchase: ", useBFchase)
    print("useImp3C: ", useImp3C)
    print("useHoloClean: ", useHoloClean)

    isSET = arg_dict["isSET"]
    is_Holoclean_or_Imp3C = False
    if useImp3C or useHoloClean:
        is_Holoclean_or_Imp3C = True

    # --------------------------------------------------- load data ---------------------------------------------------
    '''Remark: we use default_ratio=1.0 for Sigma and KG in the framework'''

    imp = Imputation()
    vary_ratio = arg_dict["vary_ratio"]
    default_ratio = arg_dict["default_ratio"]
    default_KG_ratio = arg_dict["default_KG_ratio"]
    default_training_ratio = arg_dict["default_training_ratio"]

    data_name = arg_dict["data_name"]
    GT = imp.load_Gamma(arg_dict["groundtruth"], arg_dict["varyGT"], vary_ratio, default_ratio) # normal version
    print("Groundtruth size: ", GT.shape[0])

    rees, Mc = None, None
    if useREE is True:
        rees = imp.load_Sigma(arg_dict["ree_path"], arg_dict["varyREE"], vary_ratio, default_ratio)
        print("REEs num: ", len(rees))

    Mc_model = None
    if useMc is True:
        Mc_model = CorrelationModel()
        if arg_dict["varyMc"] is True:
            Mc_model.load_model_Mc(data_name, arg_dict["cuda"], arg_dict["if_parallel"], vary_ratio, (1 - default_KG_ratio), arg_dict["Mc_type"])
        elif arg_dict["varyKG"] is True:
            Mc_model.load_model_Mc(data_name, arg_dict["cuda"], arg_dict["if_parallel"], default_training_ratio, (1.0 - vary_ratio), arg_dict["Mc_type"])
        else:
            Mc_model.load_model_Mc(data_name, arg_dict["cuda"], arg_dict["if_parallel"], default_training_ratio, (1.0 - default_KG_ratio), arg_dict["Mc_type"])
        if arg_dict["use_Mc_rules"] is True:
            Mc = load_Mc(arg_dict["Mc_path"])
            print("Mc rules num: ", len(Mc))

    Md_model = None
    if useMd is True:
        Md_model = ImputationModel()
        if arg_dict["varyMd"] is True:
            Md_model.load_model_Md(data_name, arg_dict["cuda"], arg_dict["if_parallel"], vary_ratio, (1.0 - default_KG_ratio))
        elif arg_dict["varyKG"] is True:
            Md_model.load_model_Md(data_name, arg_dict["cuda"], arg_dict["if_parallel"], default_training_ratio, (1.0 - vary_ratio))
        else:
            Md_model.load_model_Md(data_name, arg_dict["cuda"], arg_dict["if_parallel"], default_training_ratio, (1.0 - default_KG_ratio))

    her = None
    if useKG is True:
        her = HER()
        her.load_dict_for_HER(data_name, arg_dict["varyKG"], vary_ratio, default_KG_ratio)
        her.revise_DisAM_dict(data_name)
        imp.set_varyKG(arg_dict["varyKG"], vary_ratio, default_KG_ratio)  # not used

    tuples = pd.read_csv(arg_dict["tuples_path"], index_col=0, keep_default_na=True)
    track_tid = arg_dict["track_tid"]
    if track_tid is not None:  # to show a case study for processing a tuple during DS, AA, and MI
        tuples = tuples.iloc[track_tid]
        tuples = tuples.to_frame().transpose()
        print("\n...... Track tuple:\n", tuples)
    tuples.replace('||', np.nan, inplace=True)
    full_size = tuples.shape[0]
    tuples.index = range(full_size)
    if arg_dict["varyTuples"] is True:
        use_size = int(full_size * vary_ratio)
        tuples = tuples.iloc[:use_size]
    else:
        use_size = int(full_size * default_ratio)
        tuples = tuples.iloc[:use_size]

    tuples["merge_id"] = tuples["merge_id"].str.replace(",", "||")
    tuples.rename(columns={"merge_id": "id"}, inplace=True)

    merge_id_index = [set(), set()]
    indices2ids = {}
    number_attributes = None
    if "persons" == data_name:
        number_attributes = 10
    elif "imdb" == data_name:
        number_attributes = 10
    elif "dblp" == data_name:
        number_attributes = 9
    elif "college" == data_name:
        number_attributes = 8
        GT.rename(columns={"UNITID": "id"}, inplace=True)
        imp.Gamma.rename(columns={"UNITID": "id"}, inplace=True)

    tuples_to_check = tuples.iloc[:, 0:number_attributes]
    tuples_to_check = pd.concat([tuples_to_check, tuples["id"]], axis=1)
    if arg_dict["useDittoRules"] is True:
        tuples_to_check = pd.concat([tuples_to_check, tuples["confidence"]], axis=1)

    print("size of tuples to be checked: ", tuples_to_check.shape[0])

    # --------------------------------------------------- DecideTS ---------------------------------------------------
    exist_conflict, conflicting_attributes, sharp_split_indices = None, None, None
    partial_decideTS_time = 0
    if arg_dict["evaluate_separately"] is False or arg_dict["run_DecideTS"] is True or arg_dict["run_Splitting"] is True:
        print("#### start DecideTS...")
        check_sharp = True
        if useImp3C is True or useHoloClean is True or useREE is False:
            check_sharp = False
            tuples = removeSharp(tuples)
            tuples_to_check = removeSharp(tuples_to_check)
        exist_conflict_file = arg_dict["output_exist_conflict_dir"] + "partial_exist_conflict.txt"
        sharp_split_indices_file = arg_dict["output_exist_conflict_dir"] + "sharp_split_indices.txt"
        if not os.path.exists(exist_conflict_file):
            if arg_dict["count_violations"] and isSET:  # count the number of violations
                decideTS_simplified_HyperCube_new_count_violations(GT, rees, Mc, tuples_to_check, Mc_model, arg_dict["Mc_conf"], arg_dict["use_Mc_rules"], arg_dict["useDittoRules"], arg_dict["thr_ditto_rule"], tuples["label"].values)
                sys.exit(0)
            start_decideTS = timeit.default_timer()
            exist_conflict, conflicting_attributes, sharp_split_indices = decideTS_simplified_HyperCube_new(GT, rees, Mc, tuples_to_check, useREE, useMc, arg_dict["output_exist_conflict_dir"], Mc_model, check_sharp, arg_dict["Mc_conf"], arg_dict["use_Mc_rules"], arg_dict["evaluateDS_syn"], arg_dict["useDittoRules"], arg_dict["thr_ditto_rule"], arg_dict["output_results"])  # new tuples with ||
            end_decideTS = timeit.default_timer()
            partial_decideTS_time = end_decideTS - start_decideTS
            if arg_dict["output_results"] is True:
                f = open(exist_conflict_file, "a+")
                f.writelines("\nTime: " + str(partial_decideTS_time))
                f.close()
            print("#### finish partial DecideTS, using Time: ", partial_decideTS_time)
        else:
            # read exist_conflict by file
            f = open(exist_conflict_file, "r")
            lines = f.readlines()
            exist_conflict = lines[0]
            exist_conflict = [i.strip() for i in exist_conflict.split("]")[0].split("[")[1].split(",")]
            exist_conflict = list(map(int, exist_conflict))
            exist_conflict = np.array(exist_conflict)
            partial_decideTS_time = float(lines[-1].split(":")[1].strip())
            f.close()
            conflicting_attributes = None  # no use
            print("#### finish partial DecideTS, by loading results from", exist_conflict_file, ". The partial DecideTS Time: ", partial_decideTS_time)
            # read sharp_split_indices by file
            # if os.path.exists(sharp_split_indices_file):
            #     f = open(sharp_split_indices_file, "r")
            #     lines = f.readlines()
            #     sharp_split_indices = lines[0]
            #     sharp_split_indices = [i.strip() for i in sharp_split_indices.split("]")[0].split("[")[1].split(",")]
            #     sharp_split_indices = list(map(int, sharp_split_indices))
            #     sharp_split_indices = np.array(sharp_split_indices)
            #     f.close()
            sharp_split_indices = None

    print("Initial |Gamma| = ", imp.obtain_Gamma_size())

    # ---------------------------------- update Gamma, add tuples that not to split -----------------------------------
    if_update_Gamma = True  # update matched pairs to Gamma
    # if_update_Gamma = False  # do not update Gamma
    # if useImp3C is True or useHoloClean is True:  # uncomment this if updateGamma is forbidden for Holoclean and Imp3C
    #     if_update_Gamma = False

    # for decideTS: decide whether reference KG to distinguish split and repair
    reference_KG = True
    if useImp3C is True or useHoloClean is True or her is None:  # they can only repair instead of split tuples, include SET_noHER
        reference_KG = False
        conflict_indices = np.where(exist_conflict == 1)[0]
        exist_conflict[conflict_indices] = -1

    add_tuples_file = arg_dict["output_exist_conflict_dir"] + "add_tuples.csv"
    if arg_dict["evaluate_separately"] is False or arg_dict["run_Splitting"] is True:
        if useREE is True and if_update_Gamma is True:
            if not os.path.exists(add_tuples_file):
                start_updateGT = timeit.default_timer()
                add_tuples = filter_tuples_add_to_Gamma(tuples, exist_conflict, number_attributes, imp.Gamma, rees)
                if add_tuples is not None:
                    GT = imp.update_Gamma_by_tuples(add_tuples)
                    # add_tuples.to_csv(add_tuples_file, index=False)
                end_updateGT = timeit.default_timer()
                # print("#### update Gamma time:", end_updateGT - start_updateGT)
            else:
                add_tuples = pd.read_csv(add_tuples_file)
                GT = imp.update_Gamma_by_tuples(add_tuples)

    print("After DecideTS, |Gamma| = ", imp.obtain_Gamma_size())

    # -------------------- if evaluate separately, then get the ground truth exist_conflict results -------------------
    exist_conflict_groundtruth = exist_conflict
    if arg_dict["evaluate_separately"] is True:
        exist_conflict_groundtruth = copy.deepcopy(tuples["label"].values)  # 1: split; -1: repair; 0: positive
        if useImp3C is True or useHoloClean is True or her is None:  # they can only repair instead of split tuples
            conflict_indices = np.where(exist_conflict_groundtruth == 1)[0]
            exist_conflict_groundtruth[conflict_indices] = -1

    # ----------------------- if useDittoRules, then remove confidence attribute after DecideTS -----------------------
    if arg_dict["useDittoRules"] is True:
        tuples_to_check = tuples_to_check.drop(["confidence"], axis=1)

    # record the status for each attribute of each tuple
    all_attrs = tuples_to_check.columns.values.tolist()
    tuple_status = [None, None]  # 1: assign; 2: repair (i.e., set None); 3: impute; 4: repair & impute.
    for tid in [0, 1]:
        tuple_status[tid] = np.full((tuples.shape[0], len(all_attrs)), None)
        tuple_status[tid] = pd.DataFrame(tuple_status[tid], columns=all_attrs)

    # -------------------------------------- Splitting, and Attribute Assignment --------------------------------------
    splitting_tuples = [None, None]
    start_splitting, end_splitting, extra_decideTS_time = 0, 0, 0
    if arg_dict["evaluate_separately"] is False or arg_dict["run_DecideTS"] is True or arg_dict["run_Splitting"] is True:
        print("#### start splitting...")
        splitting_tuples_t0_file = arg_dict["output_tuples_splitting_dir"] + "tuples_splitting_t0.csv"
        splitting_tuples_t1_file = arg_dict["output_tuples_splitting_dir"] + "tuples_splitting_t1.csv"
        if not os.path.exists(splitting_tuples_t0_file) or not os.path.exists(splitting_tuples_t1_file):
            start_splitting = timeit.default_timer()
            if reference_KG is True:
                splitting_tuples, imputation_attr_value_indices, exist_conflict, extra_decideTS_time, tuple_status = split_HyperCube(rees, Mc, GT, tuples_to_check, exist_conflict, exist_conflict_groundtruth, sharp_split_indices, arg_dict["output_tuples_splitting_dir"], useREE, useMc, Mc_model, arg_dict["Mc_conf"], her, arg_dict["evaluate_separately"], tuple_status, arg_dict["output_results"])
            else:
                splitting_tuples, tuple_status = repair_HyperCube(rees, GT, tuples_to_check, exist_conflict_groundtruth, arg_dict["output_tuples_splitting_dir"], useREE, tuple_status, arg_dict["output_results"])  # Imp3C and Holoclean
            end_splitting = timeit.default_timer()
        else:
            splitting_tuples[0] = pd.read_csv(splitting_tuples_t0_file)
            splitting_tuples[1] = pd.read_csv(splitting_tuples_t1_file)
            # for tid in [0, 1]:
            #     splitting_tuples[tid]["id"].fillna(-1, inplace=True)
            #     splitting_tuples[tid]["id"] = splitting_tuples[tid]["id"].astype(int)
            #     splitting_tuples[tid]["id"] = splitting_tuples[tid]["id"].astype(str)
            #     splitting_tuples[tid]["id"].replace("-1", np.nan, inplace=True)
            # precision, recall, F1 = evaluate_Splitting(splitting_tuples[0], splitting_tuples[1], tuples)
            print("#### finish splitting, by loading results from", splitting_tuples_t0_file, "and", splitting_tuples_t1_file)

    if splitting_tuples is None:
        print("#### There is no tuples with conflicts")
        sys.exit(0)

    # ------------------------------------- evaluate the final decideTS results -------------------------------------
    print("#### extra decideTS time:", extra_decideTS_time)
    decideTS_time = partial_decideTS_time + extra_decideTS_time
    if arg_dict["evaluate_separately"] is False or arg_dict["run_DecideTS"] is True and track_tid is None:
        final_exist_conflict_file = arg_dict["output_exist_conflict_dir"] + "final_exist_conflict.txt"
        if not os.path.exists(final_exist_conflict_file):
            precision, recall, F1 = evaluate_DecideTS(exist_conflict, tuples["label"].values, isSET)
            if arg_dict["output_results"] is True:
                f = open(final_exist_conflict_file, "w")
                f.writelines(str(exist_conflict.tolist()))
                f.writelines("\nprecision: " + str(precision))
                f.writelines("\nrecall: " + str(recall))
                f.writelines("\nF1: " + str(F1))
                f.writelines("\nTime: " + str(decideTS_time))
                f.close()
            print("#### finish whole DecideTS, precision:", str(precision), ", recall:", str(recall), ", F1:", F1, ", using Time(partial_decideTS_time + extra_decideTS_time): ", decideTS_time)
        else:
            f = open(final_exist_conflict_file, "r")
            lines = f.readlines()
            exist_conflict = lines[0]
            exist_conflict = [i.strip() for i in exist_conflict.split("]")[0].split("[")[1].split(",")]
            exist_conflict = list(map(int, exist_conflict))
            exist_conflict = np.array(exist_conflict)
            precision = float(lines[1].split(":")[1].strip())
            recall = float(lines[2].split(":")[1].strip())
            F1 = float(lines[3].split(":")[1].strip())
            decideTS_time = float(lines[4].split(":")[1].strip())
            f.close()
            print("#### finish whole DecideTS, by loading results from", final_exist_conflict_file, ", precision:", str(precision), ", recall:", str(recall), ", F1:", F1, ", using Time(partial_decideTS_time + extra_decideTS_time): ", decideTS_time)

    if arg_dict["evaluate_separately"] is False:
        exist_conflict_groundtruth = exist_conflict  # update to the final DecideTS result

    '''
    # ------------------------------------------ evaluate Splitting results ------------------------------------------
    if arg_dict["evaluate_separately"] is False or arg_dict["run_DecideTS"] is True or arg_dict["run_Splitting"] is True:
        precision, recall, F1 = evaluate_Splitting(splitting_tuples[0], splitting_tuples[1], tuples, tuple_status, exist_conflict_groundtruth, arg_dict["evaluate_separately"], arg_dict["exist_AA"])
        print("#### finish splitting, precision:", str(precision), ", recall:", str(recall), ", F1:", F1, ", using Time: ", (end_splitting - start_splitting) - extra_decideTS_time)
    '''

    # ------------- if evaluate separately, then change splitting_tuples to be the exactly right results -------------
    splitting_tuples_groundtruth = copy.deepcopy(splitting_tuples)
    if arg_dict["evaluate_separately"] is True and arg_dict["run_Imputation"] is True:
        groundtruth_splitting_results_t0_file = arg_dict["output_tuples_splitting_dir"] + "tuples_splitting_groundtruth_t0.csv"
        groundtruth_splitting_results_t1_file = arg_dict["output_tuples_splitting_dir"] + "tuples_splitting_groundtruth_t1.csv"
        if not os.path.exists(groundtruth_splitting_results_t0_file) or not os.path.exists(groundtruth_splitting_results_t1_file):
            splitting_tuples_groundtruth = obtain_groundtruth_splitting_results(tuples, number_attributes, arg_dict["output_tuples_splitting_dir"], arg_dict["output_results"])
        else:
            splitting_tuples_groundtruth[0] = pd.read_csv(groundtruth_splitting_results_t0_file)
            splitting_tuples_groundtruth[1] = pd.read_csv(groundtruth_splitting_results_t1_file)
        # reset the status for each attribute of each tuple
        tuple_status[0].loc[:, :] = None
        tuple_status[1].loc[:, :] = None

    # -------------------------------------------------- Imputation --------------------------------------------------
    start_imputation, end_imputation = 0, 0
    tuples_imputation = [None, None]
    imputation_tuples_t0_file = arg_dict["output_tuples_imputation_dir"] + "tuples_imputation_t0.csv"
    imputation_tuples_t1_file = arg_dict["output_tuples_imputation_dir"] + "tuples_imputation_t1.csv"
    if arg_dict["evaluate_separately"] is False or arg_dict["run_Imputation"] is True or arg_dict["run_Splitting"] is True:
        print("#### start imputation...")
        if useImp3C or useHoloClean:
            pandarallel.initialize()
        # To improve performance of HoloClean
        all_tuple_size = tuples_to_check.shape[0]
        all_tuple_indices = np.array(range(all_tuple_size))
        naive_bayes_indices = []
        frequency_indices = copy.deepcopy(all_tuple_indices)
        if useHoloClean:
            hyper_ratio = 0.9
            np.random.shuffle(all_tuple_indices)
            naive_bayes_indices = all_tuple_indices[:int(all_tuple_size * hyper_ratio)]
            frequency_indices = all_tuple_indices[int(all_tuple_size * hyper_ratio):]
        if os.path.exists(imputation_tuples_t0_file) and os.path.exists(imputation_tuples_t1_file):
            print("#### Imputation has been finished!")
            sys.exit(0)
        start_imputation = timeit.default_timer()
        for tid in [0, 1]:
            # print("\n...... MI, tuple t" + str(tid) + " begin ......")  # track
            splitting_tuples_groundtruth[tid].index = np.array(range(splitting_tuples_groundtruth[tid].shape[0]))
            splitting_tuples[tid].index = np.array(range(splitting_tuples[tid].shape[0]))
            if useBFchase is False:
                tuples_imputation[tid], tuple_status[tid] = imp.chase_batch(splitting_tuples_groundtruth[tid], useREE, useKG, useMd, arg_dict["main_attribute"], exist_conflict_groundtruth, arg_dict["output_tmp_files_for_Md_dir"], her, Md_model, arg_dict["max_chase_round"], if_update_Gamma, arg_dict["impute_multi_values"], tuple_status[tid])
            else:
                all_impute_indices = np.where(exist_conflict_groundtruth != 0)[0]  # split and repair
                for index, row in splitting_tuples_groundtruth[tid].iterrows():
                    if index % 1000 == 0:
                        print("imputation for the " + str(index) + " splitting tuple")
                    if index in all_impute_indices:
                        imp.chase_all_valuations(row.to_frame().transpose(), useREE, useKG, useMd, arg_dict["max_chase_round"], her, Md_model)
            if useImp3C is True:
                tuples_imputation[tid] = imp.imputation_via_Naive_Bayes_batch(tuples_imputation[tid], exist_conflict_groundtruth)
            if useHoloClean is True:
                temp_store_1 = copy.deepcopy(tuples_imputation[tid])
                temp_store_2 = copy.deepcopy(tuples_imputation[tid])
                naive_bayes_tuples_imputation = imp.imputation_via_Naive_Bayes_batch(temp_store_1,exist_conflict_groundtruth)
                frequency_imp = imp.imputation_via_frequency(temp_store_2)
                tuples_imputation[tid].loc[naive_bayes_indices] = naive_bayes_tuples_imputation.loc[naive_bayes_indices]
                tuples_imputation[tid].loc[frequency_indices] = frequency_imp.loc[frequency_indices]

        end_imputation = timeit.default_timer()

    # ------------------------------------------ evaluate Splitting results ------------------------------------------
    evaluate = Evaluate_Splitting_and_Completing()
    evaluate.load_dict_for_evaluation(data_name)
    if arg_dict["evaluate_separately"] is False or arg_dict["run_Splitting"] is True and track_tid is None:
        precision, recall, F1 = evaluate.evaluate_Splitting(splitting_tuples[0], splitting_tuples[1], tuples_imputation[1], tuples, isSET, is_Holoclean_or_Imp3C, arg_dict["evaluate_separately"])
        print("#### finish splitting, precision:", str(precision), ", recall:", str(recall), ", F1:", F1, ", using Time: ", (end_splitting - start_splitting) - extra_decideTS_time)

    # ---------------------------------------- evaluate Imputation results ----------------------------------------
    if arg_dict["evaluate_separately"] is False or arg_dict["run_Imputation"] is True and track_tid is None:
        precision = 0
        recall = 0
        F1 = 0
        if useBFchase is True:
            precision = 'ignore'
            recall = 'ignore'
            F1 = 'ignore'
        else:
            # precision, recall, F1 = evaluate.evaluate_Completing(tuples_imputation[0], tuples_imputation[1], tuples, tuple_status, exist_conflict_groundtruth, arg_dict["evaluate_separately"], arg_dict["exist_AA"])
            precision, recall, F1 = evaluate.evaluate_Completing(tuples_imputation[0], tuples_imputation[1], tuples, isSET, is_Holoclean_or_Imp3C, arg_dict["evaluate_separately"])
        print("#### finish imputation, precision:", str(precision), ", recall:", str(recall), ", F1:", F1, ", using Time: ", end_imputation - start_imputation)

    # ----------------------------------------- write final results to files -----------------------------------------
        # imp.write_to_file(arg_dict["output_tuples_imputation"])  # this result only contains the tuples that are split and imputed

        if useBFchase is False and arg_dict["output_results"] is True:
            tuples_imputation[0].to_csv(imputation_tuples_t0_file, index=False)
            tuples_imputation[1].to_csv(imputation_tuples_t1_file, index=False)
            imp.write_groudtruth_to_file(arg_dict["output_updated_groundtrutrh"])

    print("After Completing, |Gamma| = ", imp.obtain_Gamma_size())

    # ---------------------------------------------- output running time ----------------------------------------------
    if arg_dict["evaluate_separately"] is False or (arg_dict["run_DecideTS"] is True and arg_dict["run_Splitting"] is True and arg_dict["run_Imputation"] is True):
        time_decideTS = decideTS_time
        time_splitting = end_splitting - start_splitting - extra_decideTS_time
        time_imputation = end_imputation - start_imputation
        time_whole = time_decideTS + time_splitting + time_imputation
        print('\n-------------------- The Running Time Info ----------------------------\n')
        print("The DecideTS Time: ", time_decideTS)
        print("The Splitting Time: ", time_splitting)
        print("The Imputation Time: ", time_imputation)
        print("The Entire Running Time: ", time_whole)
        print('\n-----------------------------------------------------------------------\n')
