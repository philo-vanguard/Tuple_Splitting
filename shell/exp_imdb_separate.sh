#!/bin/bash

task=(
"persons"
"imdb"
"dblp"
"college"
)


cuda=1
parallel=false
Mc_type="graph"
Mc_conf=0.3
impute_multi_values=true  # false for dblp and college; true for persons and imdb
use_Mc_rules=false
run_DecideTS=true
run_Splitting=true
run_Imputation=true
evaluateDS_syn=false
useDittoRules=true
thr_ditto_rule=0.553
default_ratio=1.0
default_KG_ratio=1.0
default_training_ratio=0.2
max_chase_round=10
evaluate_separately=true
exist_AA=true
output_results=false

for dataID in 1  # imdb
do
  echo -e "-------------------------------- SET --------------------------------"
  ./task.sh ${dataID} "varyKG" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyGT" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyREE" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

   echo -e "-------------------------------- SET_noML --------------------------------"
   ./task.sh ${dataID} "varyKG" true false true false false false false ${cuda} "SET_noML" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
   ./task.sh ${dataID} "varyGT" true false true false false false false ${cuda} "SET_noML" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
   ./task.sh ${dataID} "varyREE" true false true false false false false ${cuda} "SET_noML" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

   echo -e "-------------------------------- SET_noHER --------------------------------"
   ./task.sh ${dataID} "varyKG" true true false true false false false ${cuda} "SET_noHER" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
   ./task.sh ${dataID} "varyGT" true true false true false false false ${cuda} "SET_noHER" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
   ./task.sh ${dataID} "varyREE" true true false true false false false ${cuda} "SET_noHER" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

done
