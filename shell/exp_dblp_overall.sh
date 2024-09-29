#!/bin/bash

task=(
"persons"
"imdb"
"dblp"
"college"
)

cuda=3
parallel=false
Mc_type="graph"
Mc_conf=0.2
impute_multi_values=false
use_Mc_rules=false
run_DecideTS=true
run_Splitting=true
run_Imputation=true
evaluateDS_syn=false
useDittoRules=true
thr_ditto_rule=0.72
default_ratio=1.0
default_KG_ratio=1.0
default_training_ratio=0.2
max_chase_round=10
evaluate_separately=false
exist_AA=false
output_results=false

for dataID in 2  # dblp
do
   echo -e "-------------------------------- SET --------------------------------"
   ./task_default.sh ${dataID} "varyGT" true true true true false false false ${cuda} "SET" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

   echo -e "-------------------------------- SET_noML --------------------------------"
   ./task_default.sh ${dataID} "varyGT" true false true false false false false ${cuda} "SET_noML" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

   echo -e "-------------------------------- SET_noHER --------------------------------"
   ./task_default.sh ${dataID} "varyGT" true true false true false false false ${cuda} "SET_noHER" ${parallel} ${max_chase_round} ${default_ratio} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}

done
