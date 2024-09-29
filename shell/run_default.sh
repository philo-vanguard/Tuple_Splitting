#!/bin/bash

data_name=$1
useREE=$2
useMc=$3
useKG=$4
useMd=$5
useBFchase=$6

varyGT=$7
varyKG=$8
varyREE=$9
varyTuples=${10}
vary_ratio=${11}

useImp3C=${12}
useHoloClean=${13}

cuda=${14}

method=${15}

parallel=${16}

max_chase_round=${17}

default_ratio=${18}

varyMc=${19}
varyMd=${20}

Mc_type=${21}

evaluate_separately=${22}
Mc_conf=${23}
impute_multi_values=${24}

use_Mc_rules=${25}
run_DecideTS=${26}
run_Splitting=${27}
run_Imputation=${28}
evaluateDS_syn=${29}
useDittoRules=${30}
thr_ditto_rule=${31}

default_KG_ratio=${32}
default_training_ratio=${33}
exist_AA=${34}
output_results=${35}

main_attrs="no_use"

out_path=output
if [ "$useREE" = true ]; then
    out_path=${out_path}_useREE
fi
if [ "$useMc" = true ]; then
    out_path=${out_path}_useMc_${Mc_type}
fi
if [ "$useKG" = true ]; then
    out_path=${out_path}_useKG
fi
if [ "$useMd" = true ]; then
    out_path=${out_path}_useMd
fi
if [ "$useBFchase" = true ]; then
    out_path=${out_path}_useBFchase
fi
if [ "$varyGT" = true ]; then
    out_path=${out_path}_varyGT
fi
if [ "$varyKG" = true ]; then
    out_path=${out_path}_varyKG
fi
if [ "$varyREE" = true ]; then
    out_path=${out_path}_varyREE
fi
if [ "$varyTuples" = true ]; then
    out_path=${out_path}_varyTuples
fi
if [ "$varyMc" = true ]; then
    out_path=${out_path}_varyMc
fi
if [ "$varyMd" = true ]; then
    out_path=${out_path}_varyMd
fi
if [ "$useImp3C" = true ]; then
    out_path=${out_path}_useImp3C
fi
if [ "$useHoloClean" = true ]; then
    out_path=${out_path}_useHoloClean
fi
out_path=${out_path}_${vary_ratio}
echo -e "out_path: "${out_path}

data_dir=/tmp/tuple_splitting/datasets/
GT_path=${data_dir}${data_name}/groundtruth_0.05.csv
ree_path=${data_dir}${data_name}/logic_rules.txt
tuples_path=${data_dir}${data_name}/tuples_check.csv

output_data_dir=/tmp/tuple_splitting/results/overall/
output_results_path=${output_data_dir}${data_name}/${method}/${out_path}/
mkdir -p ${output_results_path}
exist_conflict_path=${output_results_path}
splitting_path=${output_results_path}
imputation_path=${output_results_path}
update_GT_path=${output_results_path}groundtruth_updated.csv
output_tmp_files_Md=${output_results_path}tmp_files_Md/

echo -e "--------------------------------------------------------------------------------------"
echo -e "data_name: "${data_name}
echo -e "main_attributes: "${main_attrs}
echo -e "data_dir: "${data_dir}
echo -e "GT_path: "${GT_path}
echo -e "ree_path: "${ree_path}
echo -e "tuples_path: "${tuples_path}
echo -e "out_path: "${out_path}
echo -e "exist_conflict_path: "${exist_conflict_path}
echo -e "splitting_path: "${splitting_path}
echo -e "imputation_path: "${imputation_path}
echo -e "update_GT_path: "${update_GT_path}
echo -e "output_tmp_files_Md: "${output_tmp_files_Md}
echo -e "useREE: "${useREE}
echo -e "useMc: "${useMc}
echo -e "useKG: "${useKG}
echo -e "useMd: "${useMd}
echo -e "useBFchase: "${useBFchase}
echo -e "varyGT: "${varyGT}
echo -e "varyKG: "${varyKG}
echo -e "varyREE: "${varyREE}
echo -e "varyTuples: "${varyTuples}
echo -e "varyMc: "${varyMc}
echo -e "varyMd: "${varyMd}
echo -e "vary_ratio: "${vary_ratio}
echo -e "default_ratio: "${default_ratio}
echo -e "useImp3C: "${useImp3C}
echo -e "useHoloClean: "${useHoloClean}
echo -e "cuda: "${cuda}
echo -e "method: "${method}
echo -e "parallel: "${parallel}
echo -e "max_chase_round: "${max_chase_round}
echo -e "Mc_type: "${Mc_type}
echo -e "evaluate_separately: "${evaluate_separately}
echo -e "Mc_conf: "${Mc_conf}
echo -e "impute_multi_values: "${impute_multi_values}
echo -e "use_Mc_rules: "${use_Mc_rules}
echo -e "run_DecideTS: "${run_DecideTS}
echo -e "run_Splitting: "${run_Splitting}
echo -e "run_Imputation: "${run_Imputation}
echo -e "evaluateDS_syn: "${evaluateDS_syn}
echo -e "useDittoRules: "${useDittoRules}
echo -e "thr_ditto_rule: "${thr_ditto_rule}
echo -e "default_KG_ratio: "${default_KG_ratio}
echo -e "default_training_ratio: "${default_training_ratio}
echo -e "exist_AA: "${exist_AA}
echo -e "output_results: "${output_results}
echo -e "--------------------------------------------------------------------------------------"

if [ "$method" = "SET" ]; then
  isSET=true
else
  isSET=false
fi

time=$(date "+%Y%m%d%H%M%S")
log_path=${output_results_path}log_${time}.txt

python -u ../main.py -groundtruth  ${GT_path} -ree_path ${ree_path} -tuples_path ${tuples_path} -data_name ${data_name} -main_attribute ${main_attrs} -output_exist_conflict_dir ${exist_conflict_path} -output_tuples_splitting_dir ${splitting_path} -output_tuples_imputation_dir ${imputation_path} -output_updated_groundtrutrh ${update_GT_path} -output_tmp_files_for_Md_dir ${output_tmp_files_Md} -useREE ${useREE} -useMc ${useMc} -useKG ${useKG} -useMd ${useMd} -useBFchase ${useBFchase} -varyGT ${varyGT} -varyKG ${varyKG} -varyREE ${varyREE} -varyTuples ${varyTuples} -varyMc ${varyMc} -varyMd ${varyMd} -vary_ratio ${vary_ratio} -default_ratio ${default_ratio} -useImp3C ${useImp3C} -useHoloClean ${useHoloClean} -cuda ${cuda} -if_parallel ${parallel} -max_chase_round ${max_chase_round} -Mc_type ${Mc_type} -evaluate_separately ${evaluate_separately} -Mc_conf ${Mc_conf} -impute_multi_values ${impute_multi_values} -use_Mc_rules ${use_Mc_rules} -run_DecideTS ${run_DecideTS} -run_Splitting ${run_Splitting} -run_Imputation ${run_Imputation} -evaluateDS_syn ${evaluateDS_syn} -useDittoRules ${useDittoRules} -thr_ditto_rule ${thr_ditto_rule} -default_KG_ratio ${default_KG_ratio} -default_training_ratio ${default_training_ratio} -exist_AA ${exist_AA} -output_results ${output_results} -isSET ${isSET} > ${log_path} 2>&1
