#!/bin/bash

task=(
"persons"
"imdb"
"dblp"
"college"
)

dataID=$1
expOption=$2

useREE=$3
useMc=$4
useKG=$5
useMd=$6
useBFchase=$7

useImp3C=$8
useHoloClean=$9

cuda=${10}

method=${11}

parallel=${12}

max_chase_round=${13}

default_ratio=${14}

Mc_type=${15}

evaluate_separately=${16}
Mc_conf=${17}
impute_multi_values=${18}

use_Mc_rules=${19}
run_DecideTS=${20}
run_Splitting=${21}
run_Imputation=${22}
evaluateDS_syn=${23}
useDittoRules=${24}
thr_ditto_rule=${25}

default_KG_ratio=${26}
default_training_ratio=${27}
exist_AA=${28}
output_results=${29}

varyGT=false
varyKG=false
varyREE=false
varyTuples=false
varyMc=false
varyMd=false


if [ ${expOption} = "varyGT" ]
then
  echo -e "------------------------ "${task[${dataID}]}" - vary GroundTruth ------------------------"

for vary_ratio in 0.2 0.4 0.6 0.8 1.0
do
  ./run.sh ${task[${dataID}]} ${useREE} ${useMc} ${useKG} ${useMd} ${useBFchase} true ${varyKG} ${varyREE} ${varyTuples} ${vary_ratio} ${useImp3C} ${useHoloClean} ${cuda} ${method} ${parallel} ${max_chase_round} ${default_ratio} ${varyMc} ${varyMd} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
done
fi



if [ ${expOption} = "varyKG" ]
then
  echo -e "------------------------ "${task[${dataID}]}" - vary Knowledge Graph ------------------------"

for vary_ratio in 0.2 0.4 0.6 0.8 1.0
do
  ./run.sh ${task[${dataID}]} ${useREE} ${useMc} ${useKG} ${useMd} ${useBFchase} ${varyGT} true ${varyREE} ${varyTuples} ${vary_ratio} ${useImp3C} ${useHoloClean} ${cuda} ${method} ${parallel} ${max_chase_round} ${default_ratio} ${varyMc} ${varyMd} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
done
fi



if [ ${expOption} = "varyREE" ]
then
  echo -e "------------------------ "${task[${dataID}]}" - vary REEs ------------------------"

for vary_ratio in 0.2 0.4 0.6 0.8 1.0
do
  ./run.sh ${task[${dataID}]} ${useREE} ${useMc} ${useKG} ${useMd} ${useBFchase} ${varyGT} ${varyKG} true ${varyTuples} ${vary_ratio} ${useImp3C} ${useHoloClean} ${cuda} ${method} ${parallel} ${max_chase_round} ${default_ratio} ${varyMc} ${varyMd} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
done
fi



if [ ${expOption} = "varyTuples" ]
then
  echo -e "------------------------ "${task[${dataID}]}" - vary varyTuples ------------------------"

for vary_ratio in 0.2 0.4 0.6 0.8 1.0
do
  ./run.sh ${task[${dataID}]} ${useREE} ${useMc} ${useKG} ${useMd} ${useBFchase} ${varyGT} ${varyKG} ${varyREE} true ${vary_ratio} ${useImp3C} ${useHoloClean} ${cuda} ${method} ${parallel} ${max_chase_round} ${default_ratio} ${varyMc} ${varyMd} ${Mc_type} ${evaluate_separately} ${Mc_conf} ${impute_multi_values} ${use_Mc_rules} ${run_DecideTS} ${run_Splitting} ${run_Imputation} ${evaluateDS_syn} ${useDittoRules} ${thr_ditto_rule} ${default_KG_ratio} ${default_training_ratio} ${exist_AA} ${output_results}
done
fi
