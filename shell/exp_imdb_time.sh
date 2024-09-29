#!/bin/bash

task=(
"persons"
"imdb"
"dblp"
"college"
)


cuda=2
parallel=false
Mc_type="graph"
Mc_conf=0.3
impute_multi_values=true
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

for dataID in 1
do
  echo -e "-------------------------------- SET --------------------------------"
  ./task.sh ${dataID} "varyTuples" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyKG" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyGT" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyREE" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

  echo -e "-------------------------------- SET_NC --------------------------------"
  ./task.sh ${dataID} "varyTuples" true true true true true false false ${cuda} "SET_NC" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyKG" true true true true true false false ${cuda} "SET_NC" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyGT" true true true true true false false ${cuda} "SET_NC" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
  ./task.sh ${dataID} "varyREE" true true true true true false false ${cuda} "SET_NC" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

done
