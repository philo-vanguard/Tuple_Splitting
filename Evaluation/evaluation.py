import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

all_data_dir = "/tmp/tuple_splitting/dict_model/"


# we consider as true positive the cases of correctly identify split or repair 
# 1 --> split; -1 --> repair; 0 --> no error
def evaluate_DecideTS(DecideTS_result, DecideTS_GT, is_SET):
    # index_true_GT_split = np.where(DecideTS_GT == 1)[0]
    # index_true_GT_repair = np.where(DecideTS_GT == -1)[0]
    # index_false_GT_non_split_nor_repair = np.where(DecideTS_GT == 0)[0]

    # TP = np.sum(DecideTS_result[index_true_GT_split] == 1) + np.sum(DecideTS_result[index_true_GT_repair] == -1) # split + repair
    # FP = np.sum(DecideTS_result[index_false_GT_non_split_nor_repair] != 0) + np.sum(DecideTS_result[index_true_GT_split] == -1) + np.sum(DecideTS_result[index_true_GT_repair] == 1) # no error --> split or repair; split --> repair; repair --> split
    # FN = np.sum(DecideTS_result[index_true_GT_split] == 0) + np.sum(DecideTS_result[index_true_GT_repair] == 0) # split --> no error; repair --> no error
    #
    # precision = TP * 1.0 / (TP + FP)
    # recall = TP * 1.0 / (TP + FN)
    # F1_score = 2 * recall * precision / (recall + precision)
    # print("self-defined precision:", precision, ", recall:", recall, ", F1:", F1_score)

    # print("-----------------------------------------------------------------")
    # print("--- TP:")
    # print("true split:", np.sum(DecideTS_result[index_true_GT_split] == 1))
    # print("true repair:", np.sum(DecideTS_result[index_true_GT_repair] == -1))
    # print("--- FP:")
    # print("true pos, predict to be split:", np.sum(DecideTS_result[index_false_GT_non_split_nor_repair] == 1))
    # print("true pos, predict to be repair:", np.sum(DecideTS_result[index_false_GT_non_split_nor_repair] == -1))
    # print("true split, predict to be repair:", np.sum(DecideTS_result[index_true_GT_split] == -1))
    # print("true repair, predict to be split:", np.sum(DecideTS_result[index_true_GT_repair] == 1))
    # print("--- FN:")
    # print("true split, predict to be postive:", np.sum(DecideTS_result[index_true_GT_split] == 0))
    # print("true repair, predict to be postive:", np.sum(DecideTS_result[index_true_GT_repair] == 0))
    # print("-----------------------------------------------------------------")

    # precision = precision_score(DecideTS_GT, DecideTS_result, average="weighted")
    # recall = recall_score(DecideTS_GT, DecideTS_result, average="weighted")
    # F1_score = f1_score(DecideTS_GT, DecideTS_result, average="weighted")
    # print("weighted precision:", precision, ", recall:", recall, ", f1:", F1_score)
    #
    # print("--------------------------- micro ---------------------------")
    # precision = precision_score(DecideTS_GT, DecideTS_result, average="micro")
    # recall = recall_score(DecideTS_GT, DecideTS_result, average="micro")
    # F1_score = f1_score(DecideTS_GT, DecideTS_result, average="micro")
    # print("micro precision:", precision, ", recall:", recall, ", f1:", F1_score)

    print("--------------------------- macro ---------------------------")
    precision = precision_score(DecideTS_GT, DecideTS_result, average="macro")
    recall = recall_score(DecideTS_GT, DecideTS_result, average="macro")
    F1_score = f1_score(DecideTS_GT, DecideTS_result, average="macro")
    # print("macro precision:", precision, ", recall:", recall, ", f1:", F1_score)

    #-------------------------------------------- following is for SET_split and SET_correct-----------------------------------------------------
    if is_SET is True:
        DecideTS_result_for_SET_split = [val if val == 1 else 0 for val in DecideTS_result]
        DecideTS_result_for_SET_split = np.array(DecideTS_result_for_SET_split)

        print("--------------------------- SET_split macro ---------------------------")
        precision_split = precision_score(DecideTS_GT, DecideTS_result_for_SET_split, average="macro")
        recall_split = recall_score(DecideTS_GT, DecideTS_result_for_SET_split, average="macro")
        F1_score_split = f1_score(DecideTS_GT, DecideTS_result_for_SET_split, average="macro")
        print("SET_split macro precision:", precision_split, ", recall:", recall_split, ", f1:", F1_score_split)

        DecideTS_result_for_SET_correct = [val if val == 0 else -1 for val in DecideTS_result]
        DecideTS_result_for_SET_correct = np.array(DecideTS_result_for_SET_correct)

        print("--------------------------- SET_correct macro ---------------------------")
        precision_correct = precision_score(DecideTS_GT, DecideTS_result_for_SET_correct, average="macro")
        recall_correct = recall_score(DecideTS_GT, DecideTS_result_for_SET_correct, average="macro")
        F1_score_correct = f1_score(DecideTS_GT, DecideTS_result_for_SET_correct, average="macro")
        print("SET_correct macro precision:", precision_correct, ", recall:", recall_correct, ", f1:", F1_score_correct)
    #--------------------------------------------------------------------------------------------------------------------------------------------

    return precision, recall, F1_score  # we consider macro as our standard evaluation


# we evaluate AA and MI 
class Evaluate_Splitting_and_Completing:
    def __init__(self):
        self.data_name = None
        
        # for person
        # self.data_name = None
        self.person_info_all = None
        self.item_dict = None

        # for imdb
        self.imdb_info_all = None
        self.imdb_person_dict = None

        # for dblp
        self.dblp_dict = None
        # self.dblp_DisAM_dict = None
        # self.dblp_person_name_dict_reverse_generate = None

        # for college
        # self.data_name = None
        self.college_info_all = None

    def load_dict_for_evaluation(self, data_name):
        self.data_name = data_name
        if self.data_name == "persons" and self.item_dict is None:
            self.item_dict = np.load(all_data_dir + 'dict/wikimedia/item_dict_shrink.npy', allow_pickle=True).item()
            self.person_info_all = np.load(all_data_dir + 'dict/wikimedia/person_info_all_full.npy', allow_pickle=True).item()

        elif self.data_name == "imdb":
            if self.imdb_info_all is None:
                self.imdb_info_all = np.load(all_data_dir + 'dict/IMDB/imdb_info_all.npy', allow_pickle=True).item()
            if self.imdb_person_dict is None:
                self.imdb_person_dict = np.load(all_data_dir + 'dict/IMDB/imdb_person_dict.npy', allow_pickle=True).item()

        elif self.data_name == "dblp":
            if self.dblp_dict is None:
                self.dblp_dict = np.load(all_data_dir + 'dict/DBLP/dblp_dict_t5.npy',allow_pickle=True).item()

        elif self.data_name == "college" and self.college_info_all is None: # attribute in college do not contain multiple values
            self.college_info_all = np.load(all_data_dir + 'dict/college/college_info_all.npy',allow_pickle=True).item()

################################################################Splitting####################################################################################################
    def evaluate_Splitting(self, Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately):
        if report_separately is False: # overall evaluation, thus ignored here
            return "do not apply", "do not apply", "do not apply"  
        elif self.data_name == "persons":
            return self.evaluate_Splitting_persons(Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C)
        elif self.data_name == "imdb":
            return self.evaluate_Splitting_imdb(Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C)
        elif self.data_name == "dblp":
            return self.evaluate_Splitting_dblp(Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C)
        elif self.data_name == "college":
            return self.evaluate_Splitting_college(Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C)
        
    def evaluate_Splitting_persons(self, Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C):
        TP = 0
        FP = 0
        FN = 0

        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        all_attributes = Splitting_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_split_tuple = Splitting_result_left.iloc[idx] 
            second_split_tuple = Splitting_result_right.iloc[idx]

            second_completed_tuple = Completing_result_right.iloc[idx] # for evaluating data correction

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2", or empty attribute; or attribute e.g. id
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                for attribute in all_attributes:
                    # if attribute in attribute_ignore_list:
                    #     continue
                    if attribute in non_conflicting_attributes: # non-splitting attribute don't account for evaluation, for fairness
                        continue
                    if len(str(tuple_in_GT[attribute]).split('##')) > 1: # splitting attribute containing two values divided by '##'
                        if str(tuple_in_GT[attribute]).split('##')[0] == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            FN_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]).split('##')[1] == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 1: # this attribute should be assigned to the first tuple 
                        FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]): # wrong assignment
                            FP += 1
                            FP_split += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute should be assigned to the second tuple 
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            FP += 1
                            FP_split += 1
                            FP_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 0: # this attribute should be assigned to both the first and second tuples
                        FN_correct += 1 # first assign miss for SET_correct
                        if tuple_in_GT[attribute] == first_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if tuple_in_GT[attribute] == second_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -2: # for debug purpose; this attribute should be repaired; this case should not exist 
                        print("Wrong data Format!!!")

            elif int(tuple_in_GT['label']) == -1: # an erroneous tuple to repair
                second_tuple_in_KG = self.person_info_all[int(str(tuple_in_GT['id']).split('||')[1])] # two split tuples via '||' have the same ids
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]

                    if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue

                    attribute_val_ids = str(second_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                    attribute_vals = []
                    for val_id in attribute_val_ids:
                        attribute_vals.append(str(self.item_dict[int(val_id)]))

                    if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair; impute and fill a None value 
                        FN_split += 1
                        if str(second_completed_tuple[attribute]) in attribute_vals:
                            TP += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1 
                                FP_correct += 1
            
            else: # this tuple should not be split nor repaired 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes:
                        continue
                    if str(first_split_tuple[attribute]) != str(second_split_tuple[attribute]): 
                        FP += 1
                        FP_split += 1
                        FP_correct += 1

#-------------------------------------for SET_split and SET_correct in AA phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split AA TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct AA TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct

        precision = 0
        recall = 0
        F1_score = 0
        if TP + FP == 0:
            precision = TP * 1.0 / 1.0
        else:
            precision = TP * 1.0 / (TP + FP)
        if TP + FN == 0:
            recall = TP * 1.0 / 1.0
        else:
            recall = TP * 1.0 / (TP + FN)
        if recall + precision == 0:
            F1_score = 2 * recall * precision / 1.0
        else:
            F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score
    
    def evaluate_Splitting_imdb(self, Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C):
        TP = 0
        FP = 0
        FN = 0

        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        person_relatd_attributes_for_KG = ['actor', 'actress', 'director', 'producer', 'writer']
        attribute_may_involve_float = ['startYear', 'runtimeMinutes']

        all_attributes = Splitting_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_split_tuple = Splitting_result_left.iloc[idx] 
            second_split_tuple = Splitting_result_right.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2", or empty attribute; or attribute e.g. id
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # non-splitting attribute don't account for evaluation, for fairness
                        continue
                    if len(str(tuple_in_GT[attribute]).split('##')) > 1: # splitting attribute containing two values divided by '##'
                        if str(tuple_in_GT[attribute]).split('##')[0] == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            FN_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]).split('##')[1] == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 1: # this attribute should be assigned to the first tuple 
                        FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]): # wrong assignment
                            FP += 1
                            FP_split += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute should be assigned to the second tuple 
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            FP += 1
                            FP_split += 1
                            FP_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 0: # this attribute should be assigned to both the first and second tuples
                        FN_correct += 1 # first assign miss for SET_correct
                        if tuple_in_GT[attribute] == first_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if tuple_in_GT[attribute] == second_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -2: # for debug purpose; this attribute should be repaired; this case should not exist 
                        print("Wrong data Format!!!")

            elif int(tuple_in_GT['label']) == -1: # an erroneous tuple to repair
                second_tuple_in_KG = self.imdb_info_all[str(tuple_in_GT['id']).split('||')[1]]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]
                    if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue

                    attribute_val_ids = str(second_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                    attribute_vals = []
                    if attribute in person_relatd_attributes_for_KG:
                        attribute_val_ids = str(second_attribute_value_in_KG).split(',') # for attributes referring to imdb_person_dict e.g., actor
                        for val_id in attribute_val_ids:
                            if val_id in self.imdb_person_dict.keys():
                                attribute_vals.append(str(self.imdb_person_dict[val_id]))
                    elif attribute in attribute_may_involve_float:
                        attribute_vals = str(int(float(second_attribute_value_in_KG)))
                    else: # attribute with answers in KG, e.g., 'startYear': 1905
                        # attribute_vals.append(str(second_attribute_value_in_KG))
                        attribute_vals = str(second_attribute_value_in_KG).split(',')

                    if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair
                        FN_split += 1
                        if str(second_completed_tuple[attribute]) in attribute_vals:
                            TP += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1
                                FP_correct += 1
            
            else: # this tuple should not be split or repaired 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes:
                        continue
                    if str(first_split_tuple[attribute]) != str(second_split_tuple[attribute]): 
                        FP += 1
                        FP_split += 1
                        FP_correct += 1

        #-------------------------------------for SET_split and SET_correct in AA phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split AA TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct AA TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct
                    
            
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score
        
    def evaluate_Splitting_dblp(self, Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C):
        TP = 0 
        FP = 0
        FN = 0

        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        dblp_map_dict = {'type':'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'publish':'https://dblp.org/rdf/schema#publishedIn',
        'year':'https://dblp.org/rdf/schema#yearOfPublication',
        'page':'https://dblp.org/rdf/schema#pagination',
        'author':'https://dblp.org/rdf/schema#authoredBy',
        'vol':'vol',
        'title':'https://dblp.org/rdf/schema#title',
        'ref':'http://www.w3.org/2000/01/rdf-schema#label',
        }

        all_attributes = Splitting_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_split_tuple = Splitting_result_left.iloc[idx] 
            second_split_tuple = Splitting_result_right.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2", or empty attribute; or attribute e.g. id
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # non-splitting attribute don't account for evaluation, for fairness
                        continue
                    if len(str(tuple_in_GT[attribute]).split('##')) > 1: # splitting attribute containing two values divided by '##'
                        if str(tuple_in_GT[attribute]).split('##')[0] == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            FN_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]).split('##')[1] == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 1: # this attribute should be assigned to the first tuple
                        FN_correct += 1 
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]): # wrong assignment
                            FP += 1
                            FP_split += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute should be assigned to the second tuple 
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            FP += 1
                            FP_split += 1
                            FP_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 0: # this attribute should be assigned to both the first and second tuples
                        FN_correct += 1 # first assign miss for SET_correct
                        if tuple_in_GT[attribute] == first_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if tuple_in_GT[attribute] == second_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -2: # for debug purpose; this attribute should be repaired; this case should not exist 
                        print("Wrong data Format!!!")

            elif int(tuple_in_GT['label']) == -1: # an erroneous tuple to repair
                tuple_in_KG = self.dblp_dict[str(tuple_in_GT['id']).split('||')[1]]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue

                    attribute_value_in_KG = ''
                    if attribute == "author_1":
                        if dblp_map_dict['author'] in tuple_in_KG.keys():
                            attribute_value_in_KG =  tuple_in_KG[dblp_map_dict['author']].split('||')[0]
                        else:
                            attribute_value_in_KG = ''
                    elif attribute == "author_2":
                        if dblp_map_dict['author'] in tuple_in_KG.keys():
                            attribute_value_in_KG =  [tuple_in_KG[dblp_map_dict['author']].split('||')[1] if len(tuple_in_KG[dblp_map_dict['author']].split('||')) > 1 else '']
                            attribute_value_in_KG = attribute_value_in_KG[0] # list --> value
                        else:
                            attribute_value_in_KG = ''
                    else:
                        attribute_value_in_KG = ''
                        if dblp_map_dict[str(attribute).replace("_", " ")] in tuple_in_KG.keys():
                            tuple_in_KG[dblp_map_dict[str(attribute).replace("_", " ")]]
                    
                    if attribute_value_in_KG == '' or pd.isnull(attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue
                    else:
                        attribute_vals = []
                        if attribute == "vol":
                            attribute_vals = attribute_value_in_KG
                        else:
                            attribute_vals = attribute_value_in_KG.split('||')

                        if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair
                            FN_split += 1
                            if str(second_completed_tuple[attribute]) in attribute_vals:
                                TP += 1
                                TP_correct += 1
                            else:
                                FN += 1
                                FN_correct += 1
                                if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1
                                    FP_correct += 1
            
            else: # this tuple should not be split or repaired 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes:
                        continue
                    if str(first_split_tuple[attribute]) != str(second_split_tuple[attribute]): 
                        FP += 1
                        FP_split += 1
                        FP_correct += 1
                    
        #-------------------------------------for SET_split and SET_correct in AA phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split AA TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct AA TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct
            
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score
        
    def evaluate_Splitting_college(self, Splitting_result_left, Splitting_result_right, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C):
        TP = 0 
        FP = 0
        FN = 0

        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        all_attributes = Splitting_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_split_tuple = Splitting_result_left.iloc[idx] 
            second_split_tuple = Splitting_result_right.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2", or empty attribute; or attribute e.g. id
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # non-splitting attribute don't account for evaluation, for fairness
                        continue
                    if len(str(tuple_in_GT[attribute]).split('##')) > 1: # splitting attribute containing two values divided by '##'
                        if str(tuple_in_GT[attribute]).split('##')[0] == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            FN_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]).split('##')[1] == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 1: # this attribute should be assigned to the first tuple 
                        FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]): # wrong assignment
                            FP += 1
                            FP_split += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute should be assigned to the second tuple 
                        if str(tuple_in_GT[attribute]) == str(second_split_tuple[attribute]):
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                        if str(tuple_in_GT[attribute]) == str(first_split_tuple[attribute]):
                            FP += 1
                            FP_split += 1
                            FP_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == 0: # this attribute should be assigned to both the first and second tuples
                        FN_correct += 1 # first assign miss for SET_correct
                        if tuple_in_GT[attribute] == first_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                        if tuple_in_GT[attribute] == second_split_tuple[attribute]:
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                    elif int(tuple_in_GT["label_"+attribute]) == -2: # for debug purpose; this attribute should be repaired; this case should not exist 
                        print("Wrong data Format!!!")

            elif int(tuple_in_GT['label']) == -1: # an erroneous tuple to repair
                second_tuple_in_KG = self.college_info_all[int(str(tuple_in_GT['id']).split('||')[1])]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]
                    if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue

                    if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair
                        FN_split += 1
                        if str(second_completed_tuple[attribute]) == str(second_attribute_value_in_KG): # KG has the ground-truth
                            TP += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1
                                FP_correct += 1
            
            else: # this tuple should not be split or repaired 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes:
                        continue
                    if str(first_split_tuple[attribute]) != str(second_split_tuple[attribute]): 
                        FP += 1
                        FP_split += 1
                        FP_correct += 1

        #-------------------------------------for SET_split and SET_correct in AA phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split AA TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct AA TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct AA precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct
                    
            
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score


################################################################Splitting####################################################################################################



################################################################Completing####################################################################################################
    def evaluate_Completing(self, Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately):
        if self.data_name == "persons":
            return self.evaluate_Completing_persons(Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately)
        elif self.data_name == "imdb":
            return self.evaluate_Completing_imdb(Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately)
        elif self.data_name == "dblp":
            return self.evaluate_Completing_dblp(Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately)
        elif self.data_name == "college":
            return self.evaluate_Completing_college(Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately)


    # Completing_result_left.size : Completing_result_right.size : merge_result_GT.size = 1 : 1 : 1;
    # attribute_ignore_list (expired) --> attribute can be ignored e.g., 'id'
    # entity_dict --> person_info_all in wiki; attribute_dict --> item_dict
    # def evaluate_Completing(Completing_result_left, Completing_result_right, merge_result_GT, attribute_ignore_list, entity_dict, attribute_dict):
    def evaluate_Completing_persons(self, Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        all_attributes = Completing_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_completed_tuple = Completing_result_left.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or len(str(tuple_in_GT[attribute]).split('##')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2"; and splitting duplicate attribute contains values, e.g., "c1##c2"; empty attributes; attribute e.g. 'id'
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                first_tuple_in_KG = self.person_info_all[int(str(tuple_in_GT['id']).split('||')[0])]
                second_tuple_in_KG = self.person_info_all[int(str(tuple_in_GT['id']).split('||')[1])]
                for attribute in all_attributes:
                    # if attribute in attribute_ignore_list: # ignore attributes e.g., id
                    #     continue
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    first_attribute_value_in_KG =  first_tuple_in_KG[str(attribute).replace("_", " ")]
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]

                    if report_separately is False: # overall evaluation
                        if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                            continue
                        else:
                            attribute_val_ids = str(first_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                            attribute_vals = []
                            for val_id in attribute_val_ids:
                                attribute_vals.append(str(self.item_dict[int(val_id)]))

                            FN_correct += 1 # cannot deal with the values in the first tuple
                            
                            if str(first_completed_tuple[attribute]) in attribute_vals:
                                TP += 1
                                TP_split += 1
                            else:
                                FN += 1
                                FN_split += 1
                                if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced falsely
                                    FP += 1
                                    FP_split += 1

                        if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                        else: # KG has the ground-truth
                            attribute_val_ids = str(second_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                            attribute_vals = []
                            for val_id in attribute_val_ids:
                                attribute_vals.append(str(self.item_dict[int(val_id)]))
                            if str(second_completed_tuple[attribute]) in attribute_vals:
                                TP += 1
                                TP_split += 1
                                TP_correct += 1
                            else:
                                FN += 1
                                FN_split += 1
                                FN_correct += 1
                                if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced falsely
                                    FP += 1
                                    FP_split += 1
                                    FP_correct += 1

                    else: # separate evaluation
                        if int(tuple_in_GT["label_"+attribute]) == 1: # this attribute of the second tuple need to be imputed, since it is assigned to the first tuple
                            if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            else: # KG has the ground-truth
                                attribute_val_ids = str(second_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                                attribute_vals = []
                                for val_id in attribute_val_ids:
                                    attribute_vals.append(str(self.item_dict[int(val_id)]))
                                if str(second_completed_tuple[attribute]) in attribute_vals:
                                    TP += 1
                                    TP_split += 1
                                    TP_correct += 1
                                else:
                                    FN += 1
                                    FN_split += 1
                                    FN_correct += 1
                                    if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                        FP += 1
                                        FP_split += 1
                                        FP_correct += 1
                        elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute of the first tuple need to be imputed, since it is assigned to the second tuple
                            if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            else:
                                FN_correct += 1
                                attribute_val_ids = str(first_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                                attribute_vals = []
                                for val_id in attribute_val_ids:
                                    attribute_vals.append(str(self.item_dict[int(val_id)]))
                                if str(first_completed_tuple[attribute]) in attribute_vals:
                                    TP += 1
                                    TP_split += 1
                                else:
                                    FN += 1
                                    FN_split += 1
                                    if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                        FP += 1
                                        FP_split += 1

            elif int(tuple_in_GT['label']) == -1: # a tuple needs to repair without split
                if report_separately is True: # corrected in AA phase, thus ignored
                    continue

                FP_split += 2

                second_tuple_in_KG = self.person_info_all[int(str(tuple_in_GT['id']).split('||')[1])] # two split tuples via '||' have the same ids
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]

                    if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue

                    attribute_val_ids = str(second_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                    attribute_vals = []
                    for val_id in attribute_val_ids:
                        attribute_vals.append(str(self.item_dict[int(val_id)]))

                    if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair; impute and fill a None value 
                        FN_split += 1
                        if str(second_completed_tuple[attribute]) in attribute_vals:
                            TP += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1 
                                FP_correct += 1

            else: # wrongly split or correct a tuple 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue

                    if str(first_completed_tuple[attribute]) != str(second_completed_tuple[attribute]): # differ due to splitting
                        FP += 1
                        FP_split += 1
                        FP_correct += 1

        #-------------------------------------for SET_split and SET_correct in MI phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split MI TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct MI TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct

        precision = 0
        recall = 0
        F1_score = 0
        if TP + FP == 0:
            precision = TP * 1.0 / 1.0
        else:
            precision = TP * 1.0 / (TP + FP)
        if TP + FN == 0:
            recall = TP * 1.0 / 1.0
        else:
            recall = TP * 1.0 / (TP + FN)
        if recall + precision == 0:
            F1_score = 2 * recall * precision / 1.0
        else:
            F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score

    def evaluate_Completing_imdb(self, Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        person_relatd_attributes_for_KG = ['actor', 'actress', 'director', 'producer', 'writer']
        attribute_may_involve_float = ['startYear', 'runtimeMinutes']
        # attributes_with_vertical_line_for_KG = ['title']

        all_attributes = Completing_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_completed_tuple = Completing_result_left.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or len(str(tuple_in_GT[attribute]).split('##')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2"; and splitting duplicate attribute contains values, e.g., "c1##c2"; empty attributes; attribute e.g. 'id'
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                first_tuple_in_KG = self.imdb_info_all[str(tuple_in_GT['id']).split('||')[0]]
                second_tuple_in_KG = self.imdb_info_all[str(tuple_in_GT['id']).split('||')[1]]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    first_attribute_value_in_KG =  first_tuple_in_KG[str(attribute).replace("_", " ")]
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]

                    if report_separately is False: # overall evaluation
                        if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                            continue
                        else:
                            attribute_vals = []
                            if attribute in person_relatd_attributes_for_KG:
                                attribute_val_ids = str(first_attribute_value_in_KG).split(',') # for attributes referring to imdb_person_dict e.g., actor
                                for val_id in attribute_val_ids:
                                    if val_id in self.imdb_person_dict.keys():
                                        attribute_vals.append(str(self.imdb_person_dict[val_id]))
                            elif attribute in attribute_may_involve_float:
                                attribute_vals = str(int(float(first_attribute_value_in_KG)))
                            else: # attribute with answers in KG, e.g., 'startYear': 1905
                                    # attribute_vals.append(str(first_attribute_value_in_KG))
                                attribute_vals = str(first_attribute_value_in_KG).split(',')

                            FN_correct += 1 # cannot deal with the values in the first tuple
                            if str(first_completed_tuple[attribute]) in attribute_vals:
                                TP += 1
                                TP_split += 1
                            else:
                                FN += 1
                                FN_split += 1
                                if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1
                                    FP_split += 1

                        if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                            continue
                        else: # KG has the ground-truth
                            attribute_vals = []
                            if attribute in person_relatd_attributes_for_KG:
                                attribute_val_ids = str(second_attribute_value_in_KG).split(',') # for attributes referring to imdb_person_dict e.g., actor
                                for val_id in attribute_val_ids:
                                    if val_id in self.imdb_person_dict.keys():
                                        attribute_vals.append(str(self.imdb_person_dict[val_id]))
                            elif attribute in attribute_may_involve_float:
                                attribute_vals = str(int(float(second_attribute_value_in_KG)))
                            else: # attribute with answers in KG, e.g., 'startYear': 1905
                                # attribute_vals.append(str(second_attribute_value_in_KG))
                                attribute_vals = str(second_attribute_value_in_KG).split(',')
                            if str(second_completed_tuple[attribute]) in attribute_vals:
                                TP += 1
                                TP_split += 1
                                TP_correct += 1
                            else:
                                FN += 1
                                FN_split += 1
                                FN_correct += 1
                                if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1   
                                    FP_split += 1
                                    FP_correct += 1

                    else: # separate evaluation
                        if int(tuple_in_GT["label_"+attribute]) == 1: # this attribute of the second tuple need to be imputed, since it is assigned to the first tuple
                            if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            else: # KG has the ground-truth
                                attribute_vals = []
                                if attribute in person_relatd_attributes_for_KG:
                                    attribute_val_ids = str(second_attribute_value_in_KG).split(',') # for attributes referring to imdb_person_dict e.g., actor
                                    for val_id in attribute_val_ids:
                                        if val_id in self.imdb_person_dict.keys():
                                            attribute_vals.append(str(self.imdb_person_dict[val_id]))
                                elif attribute in attribute_may_involve_float:
                                    attribute_vals = str(int(float(second_attribute_value_in_KG)))
                                else: # attribute with answers in KG, e.g., 'startYear': 1905
                                    # attribute_vals.append(str(second_attribute_value_in_KG))
                                    attribute_vals = str(second_attribute_value_in_KG).split(',')
                                if str(second_completed_tuple[attribute]) in attribute_vals:
                                    TP += 1
                                    TP_split += 1
                                    TP_correct += 1
                                else:
                                    FN += 1
                                    FN_split += 1
                                    FN_correct += 1
                                    if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                        FP += 1
                                        FP_split += 1
                                        FP_correct += 1
                        
                        elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute of the first tuple need to be imputed, since it is assigned to the second tuple
                            if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            else:
                                FN_correct += 1
                                attribute_vals = []
                                if attribute in person_relatd_attributes_for_KG:
                                    attribute_val_ids = str(first_attribute_value_in_KG).split(',') # for attributes referring to imdb_person_dict e.g., actor
                                    for val_id in attribute_val_ids:
                                        if val_id in self.imdb_person_dict.keys():
                                            attribute_vals.append(str(self.imdb_person_dict[val_id]))
                                elif attribute in attribute_may_involve_float:
                                    attribute_vals = str(int(float(first_attribute_value_in_KG)))
                                else: # attribute with answers in KG, e.g., 'startYear': 1905
                                    # attribute_vals.append(str(first_attribute_value_in_KG))
                                    attribute_vals = str(first_attribute_value_in_KG).split(',')
                                if str(first_completed_tuple[attribute]) in attribute_vals:
                                    TP += 1
                                    TP_split += 1
                                else:
                                    FN += 1
                                    FN_split += 1
                                    if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                        FP += 1
                                        FP_split += 1
            
            elif int(tuple_in_GT['label']) == -1: # a tuple needs to repair without split
                if report_separately is True: # corrected in AA phase, thus ignored
                    continue

                FP_split += 2

                second_tuple_in_KG = self.imdb_info_all[str(tuple_in_GT['id']).split('||')[1]]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]
                    if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue

                    attribute_val_ids = str(second_attribute_value_in_KG).split(',') # attribute may include multiple values, e.g., values for attributes award and member of
                    attribute_vals = []
                    if attribute in person_relatd_attributes_for_KG:
                        attribute_val_ids = str(second_attribute_value_in_KG).split(',') # for attributes referring to imdb_person_dict e.g., actor
                        for val_id in attribute_val_ids:
                            if val_id in self.imdb_person_dict.keys():
                                attribute_vals.append(str(self.imdb_person_dict[val_id]))
                    elif attribute in attribute_may_involve_float:
                        attribute_vals = str(int(float(second_attribute_value_in_KG)))
                    else: # attribute with answers in KG, e.g., 'startYear': 1905
                        # attribute_vals.append(str(second_attribute_value_in_KG))
                        attribute_vals = str(second_attribute_value_in_KG).split(',')

                    if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair
                        FN_split += 1
                        if str(second_completed_tuple[attribute]) in attribute_vals:
                            TP += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1 
                                FP_correct += 1

            else: # wrongly split or correct a tuple 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    if str(first_completed_tuple[attribute]) != str(second_completed_tuple[attribute]): # differ due to splitting
                        FP += 1          
                        FP_split += 1
                        FP_correct += 1  

        #-------------------------------------for SET_split and SET_correct in MI phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split MI TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct MI TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct

        precision = 0
        recall = 0
        F1_score = 0
        if TP + FP == 0:
            precision = TP * 1.0 / 1.0
        else:
            precision = TP * 1.0 / (TP + FP)
        if TP + FN == 0:
            recall = TP * 1.0 / 1.0
        else:
            recall = TP * 1.0 / (TP + FN)
        if recall + precision == 0:
            F1_score = 2 * recall * precision / 1.0
        else:
            F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score

    def evaluate_Completing_dblp(self, Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately):
        dblp_map_dict = {'type':'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'publish':'https://dblp.org/rdf/schema#publishedIn',
        'year':'https://dblp.org/rdf/schema#yearOfPublication',
        'page':'https://dblp.org/rdf/schema#pagination',
        'author':'https://dblp.org/rdf/schema#authoredBy',
        'vol':'vol',
        'title':'https://dblp.org/rdf/schema#title',
        'ref':'http://www.w3.org/2000/01/rdf-schema#label',
        }

        TP = 0
        FP = 0
        FN = 0
        TN = 0
        # id_attributes_for_KG = ['type', 'author'] 

        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C

        all_attributes = Completing_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_completed_tuple = Completing_result_left.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or len(str(tuple_in_GT[attribute]).split('##')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2"; and splitting duplicate attribute contains values, e.g., "c1##c2"; empty attributes; attribute e.g. 'id'
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                first_tuple_in_KG = self.dblp_dict[str(tuple_in_GT['id']).split('||')[0]]
                second_tuple_in_KG = self.dblp_dict[str(tuple_in_GT['id']).split('||')[1]]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    first_attribute_value_in_KG = ''
                    second_attribute_value_in_KG = ''
                    if attribute == "author_1":
                        if dblp_map_dict['author'] in first_tuple_in_KG.keys():
                            first_attribute_value_in_KG =  first_tuple_in_KG[dblp_map_dict['author']].split('||')[0]
                        else: # the tuple in KG does not have this attribute
                            first_attribute_value_in_KG =  ''
                        if dblp_map_dict['author'] in second_tuple_in_KG.keys():
                            second_attribute_value_in_KG =  second_tuple_in_KG[dblp_map_dict['author']].split('||')[0]
                        else:
                            second_attribute_value_in_KG = ''
                    elif attribute == "author_2":
                        if dblp_map_dict['author'] in first_tuple_in_KG.keys():
                            first_attribute_value_in_KG =  [first_tuple_in_KG[dblp_map_dict['author']].split('||')[1] if len(first_tuple_in_KG[dblp_map_dict['author']].split('||')) > 1 else '']
                            first_attribute_value_in_KG = first_attribute_value_in_KG[0] # list --> value
                        else: # the tuple in KG does not have this attribute
                            first_attribute_value_in_KG =  ''
                        if dblp_map_dict['author'] in second_tuple_in_KG.keys():
                            second_attribute_value_in_KG =  [second_tuple_in_KG[dblp_map_dict['author']].split('||')[1] if len(second_tuple_in_KG[dblp_map_dict['author']].split('||')) > 1 else '']
                            second_attribute_value_in_KG = second_attribute_value_in_KG[0] # list --> value
                        else:
                            second_attribute_value_in_KG = ''
                    else:
                        # first_attribute_value_in_KG =  ''
                        if dblp_map_dict[str(attribute).replace("_", " ")] in first_tuple_in_KG.keys():
                            first_attribute_value_in_KG = first_tuple_in_KG[dblp_map_dict[str(attribute).replace("_", " ")]]
                        # second_attribute_value_in_KG = ''
                        if dblp_map_dict[str(attribute).replace("_", " ")] in second_tuple_in_KG.keys():
                            second_attribute_value_in_KG = second_tuple_in_KG[dblp_map_dict[str(attribute).replace("_", " ")]]

                    if report_separately is False: # overall evaluation
                        if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                            continue
                        else:
                            attribute_vals = []
                            if attribute == "vol":
                                attribute_vals = first_attribute_value_in_KG
                            else:
                                attribute_vals = first_attribute_value_in_KG.split('||')
                            # if first_completed_tuple[attribute].replace('"','') in attribute_vals:
                            FN_correct += 1 # cannot deal with the values in the first tuple
                            if str(first_completed_tuple[attribute]).replace("^^", "||") in attribute_vals:
                                TP += 1
                                TP_split += 1
                            else:
                                FN += 1
                                FN_split += 1
                                if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1
                                    FP_split += 1

                        if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                            continue
                        else: # KG has the ground-truth
                            attribute_vals = []
                            if attribute == "vol":
                                attribute_vals = second_attribute_value_in_KG
                            else:
                                attribute_vals = second_attribute_value_in_KG.split('||')
                            # if second_completed_tuple[attribute].replace('"','') in attribute_vals:
                            if str(second_completed_tuple[attribute]).replace("^^", "||") in attribute_vals:
                                TP += 1
                                TP_split += 1
                                TP_correct += 1
                            else:
                                FN += 1
                                FN_split += 1
                                FN_correct += 1
                                if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1
                                    FP_split += 1
                                    FP_correct += 1
                    else: # separate evaluation
                        if int(tuple_in_GT["label_"+attribute]) == 1: # this attribute of the second tuple need to be imputed, since it is assigned to the first tuple
                            if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            else: # KG has the ground-truth
                                attribute_vals = []
                                if attribute == "vol":
                                    attribute_vals = second_attribute_value_in_KG
                                else:
                                    attribute_vals = second_attribute_value_in_KG.split('||')
                                # if second_completed_tuple[attribute].replace('"','') in attribute_vals:
                                if str(second_completed_tuple[attribute]).replace("^^", "||") in attribute_vals:
                                    TP += 1
                                    TP_split += 1
                                    TP_correct += 1
                                else:
                                    FN += 1
                                    FN_split += 1
                                    FN_correct += 1
                                    if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                        FP += 1
                                        FP_split += 1
                                        FP_correct += 1
                        elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute of the first tuple need to be imputed, since it is assigned to the second tuple
                            if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            else:
                                FN_correct += 1
                                attribute_vals = []
                                if attribute == "vol":
                                    attribute_vals = first_attribute_value_in_KG
                                else:
                                    attribute_vals = first_attribute_value_in_KG.split('||')
                                # if first_completed_tuple[attribute].replace('"','') in attribute_vals:
                                if str(first_completed_tuple[attribute]).replace("^^", "||") in attribute_vals:
                                    TP += 1
                                    TP_split += 1
                                else:
                                    FN += 1
                                    FN_split += 1
                                    if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                        FP += 1
                                        FP_split += 1

            elif int(tuple_in_GT['label']) == -1: # a tuple needs to repair without split
                if report_separately is True: # corrected in AA phase, thus ignored
                    continue

                FP_split += 2

                tuple_in_KG = self.dblp_dict[str(tuple_in_GT['id']).split('||')[1]]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue

                    attribute_value_in_KG = ''
                    if attribute == "author_1":
                        if dblp_map_dict['author'] in tuple_in_KG.keys():
                            attribute_value_in_KG =  tuple_in_KG[dblp_map_dict['author']].split('||')[0]
                        else:
                            attribute_value_in_KG = ''
                    elif attribute == "author_2":
                        if dblp_map_dict['author'] in tuple_in_KG.keys():
                            attribute_value_in_KG =  [tuple_in_KG[dblp_map_dict['author']].split('||')[1] if len(tuple_in_KG[dblp_map_dict['author']].split('||')) > 1 else '']
                            attribute_value_in_KG = attribute_value_in_KG[0] # list --> value
                        else:
                            attribute_value_in_KG = ''
                    else:
                        attribute_value_in_KG = ''
                        if dblp_map_dict[str(attribute).replace("_", " ")] in tuple_in_KG.keys():
                            tuple_in_KG[dblp_map_dict[str(attribute).replace("_", " ")]]
                    
                    if attribute_value_in_KG == '' or pd.isnull(attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue
                    else:
                        attribute_vals = []
                        if attribute == "vol":
                            attribute_vals = attribute_value_in_KG
                        else:
                            attribute_vals = attribute_value_in_KG.split('||')

                        if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair
                            FN_split += 1
                            if str(second_completed_tuple[attribute]) in attribute_vals:
                                TP += 1
                                TP_correct += 1
                            else:
                                FN += 1
                                FN_correct += 1
                                if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1 
                                    FP_correct += 1

            else: # wrongly split or repair a tuple 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    if str(first_completed_tuple[attribute]) != str(second_completed_tuple[attribute]): # differ due to splitting
                        FP += 1   
                        FP_split += 1
                        FP_correct += 1

        #-------------------------------------for SET_split and SET_correct in MI phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split MI TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct MI TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct

        precision = 0
        recall = 0
        F1_score = 0
        if TP + FP == 0:
            precision = TP * 1.0 / 1.0
        else:
            precision = TP * 1.0 / (TP + FP)
        if TP + FN == 0:
            recall = TP * 1.0 / 1.0
        else:
            recall = TP * 1.0 / (TP + FN)
        if recall + precision == 0:
            F1_score = 2 * recall * precision / 1.0
        else:
            F1_score = 2 * recall * precision / (recall + precision)

        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score

    def evaluate_Completing_college(self, Completing_result_left, Completing_result_right, merge_result_GT, is_SET, is_Holoclean_or_Imp3C, report_separately):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        TP_split = 0; FP_split = 0; FN_split = 0 # for SET_split
        TP_correct = 0; FP_correct = 0; FN_correct = 0 # for SET_repair, Holoclean and Imp3C
        
        all_attributes = Completing_result_left.columns
        for idx in range(len(merge_result_GT)):
            tuple_in_GT = merge_result_GT.iloc[idx]
            first_completed_tuple = Completing_result_left.iloc[idx]
            second_completed_tuple = Completing_result_right.iloc[idx]

            non_conflicting_attributes = []
            for attribute in all_attributes:
                if len(str(tuple_in_GT[attribute]).split('||')) > 1 or len(str(tuple_in_GT[attribute]).split('##')) > 1 or pd.isnull(tuple_in_GT[attribute]) or attribute == 'id': # non-splitting attribute contains values e.g., "a1||a2"; and splitting duplicate attribute contains values, e.g., "c1##c2"; empty attributes; attribute e.g. 'id'
                    non_conflicting_attributes.append(attribute)

            if int(tuple_in_GT['label']) == 1: # a tuple of two mismatched ones
                first_tuple_in_KG = self.college_info_all[int(str(tuple_in_GT['id']).split('||')[0])]
                second_tuple_in_KG = self.college_info_all[int(str(tuple_in_GT['id']).split('||')[1])]
                for attribute in all_attributes:
                    # if attribute in attribute_ignore_list: # ignore attributes e.g., id
                    #     continue
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    first_attribute_value_in_KG =  first_tuple_in_KG[str(attribute).replace("_", " ")]
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]

                    if report_separately is False: # overall evaluation
                        if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                            continue
                        FN_correct += 1 # cannot deal with the values in the first tuple
                        if str(first_completed_tuple[attribute]) == str(first_attribute_value_in_KG):
                            TP += 1
                            TP_split += 1
                        else:
                            FN += 1
                            FN_split += 1
                            if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1
                                FP_split += 1

                        if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                            continue
                        if str(second_completed_tuple[attribute]) == str(second_attribute_value_in_KG): # KG has the ground-truth
                            TP += 1
                            TP_split += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_split += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1    
                                FP_split += 1
                                FP_correct += 1             
                    else:  # separate evaluation   
                        if int(tuple_in_GT["label_"+attribute]) == 1: # this attribute of the second tuple need to be imputed, since it is assigned to the first tuple
                            if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            elif str(second_completed_tuple[attribute]) == str(second_attribute_value_in_KG): # KG has the ground-truth
                                TP += 1
                                TP_split += 1
                                TP_correct += 1
                            else:
                                FN += 1
                                FN_split += 1
                                FN_correct += 1
                                if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1
                                    FP_split += 1
                                    FP_correct += 1
                        elif int(tuple_in_GT["label_"+attribute]) == -1: # this attribute of the first tuple need to be imputed, since it is assigned to the second tuple
                            if first_attribute_value_in_KG == '' or pd.isnull(first_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as sucess
                                continue
                                # TP += 1
                            FN_correct += 1
                            if str(first_completed_tuple[attribute]) == str(first_attribute_value_in_KG):
                                TP += 1
                                TP_split += 1
                            else:
                                FN += 1
                                FN_split += 1
                                if pd.notnull(str(first_completed_tuple[attribute])): # if this value is deduced false
                                    FP += 1 
                                    FP_split += 1

            elif int(tuple_in_GT['label']) == -1: # a tuple needs to repair without split
                if report_separately is True: # corrected in AA phase, thus ignored
                    continue

                FP_split += 2

                second_tuple_in_KG = self.college_info_all[int(str(tuple_in_GT['id']).split('||')[1])]
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    
                    second_attribute_value_in_KG =  second_tuple_in_KG[str(attribute).replace("_", " ")]
                    if second_attribute_value_in_KG == '' or pd.isnull(second_attribute_value_in_KG): # this value in KG is missing, so any filled value can be regarded as success
                        continue

                    if int(tuple_in_GT["label_"+attribute]) == -2: # attribute need to repair
                        FN_split += 1
                        if str(second_completed_tuple[attribute]) == str(second_attribute_value_in_KG): # KG has the ground-truth
                            TP += 1
                            TP_correct += 1
                        else:
                            FN += 1
                            FN_correct += 1
                            if pd.notnull(str(second_completed_tuple[attribute])): # if this value is deduced false
                                FP += 1    
                                FP_correct += 1  

            else: # wrongly split or repair a tuple 
                for attribute in all_attributes:
                    if attribute in non_conflicting_attributes: # ignore the calculation for fairness
                        continue
                    if str(first_completed_tuple[attribute]) != str(second_completed_tuple[attribute]): # differ due to splitting
                        FP += 1
                        FP_split += 1
                        FP_correct += 1

        #-------------------------------------for SET_split and SET_correct in MI phase----------------------------------------------------
        if is_SET is True:
            precision = 0
            recall = 0
            F1_score = 0
            if TP_split + FP_split == 0:
                precision = TP_split * 1.0 / 1.0
            else:
                precision = TP_split * 1.0 / (TP_split + FP_split)
            if TP_split + FN_split == 0:
                recall = TP_split * 1.0 / 1.0
            else:
                recall = TP_split * 1.0 / (TP_split + FN_split)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_split MI TP:", TP_split, ", FP:", FP_split, ", FN:", FN_split)
            print("SET_split MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#------------------------------------------------------------------------------------
            precision = 0
            recall = 0
            F1_score = 0
            if TP_correct + FP_correct == 0:
                precision = TP_correct * 1.0 / 1.0
            else:
                precision = TP_correct * 1.0 / (TP_correct + FP_correct)
            if TP_correct + FN_correct == 0:
                recall = TP_correct * 1.0 / 1.0
            else:
                recall = TP_correct * 1.0 / (TP_correct + FN_correct)
            if recall + precision == 0:
                F1_score = 2 * recall * precision / 1.0
            else:
                F1_score = 2 * recall * precision / (recall + precision)

            print("SET_correct MI TP:", TP_correct, ", FP:", FP_correct, ", FN:", FN_correct)
            print("SET_correct MI precision:", precision, ", recall:", recall, ", F1:", F1_score)
#----------------------------------------------------------------------------------------------------------------------

        if is_Holoclean_or_Imp3C is True:
            TP = TP_correct
            FP = FP_correct
            FN = FN_correct

        precision = 0
        recall = 0
        F1_score = 0
        if TP + FP == 0:
            precision = TP * 1.0 / 1.0
        else:
            precision = TP * 1.0 / (TP + FP)
        if TP + FN == 0:
            recall = TP * 1.0 / 1.0
        else:
            recall = TP * 1.0 / (TP + FN)
        if recall + precision == 0:
            F1_score = 2 * recall * precision / 1.0
        else:
            F1_score = 2 * recall * precision / (recall + precision)
        
        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)

        return precision, recall, F1_score
################################################################Completing####################################################################################################
