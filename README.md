# Tuple Splitting
There has been a host of work on entity resolution (ER), to identify tuples that refer to the same entity. This paper studies the inverse of ER, to identify tuples to which distinct real-world entities are matched by mistake, and split such tuples into a set of tuples, one for each entity. We formulate the tuple splitting problem. We propose a scheme to decide what tuples to split and what tuples to correct without splitting, fix errors/assign attribute values to the split tuples, and impute missing values. The scheme introduces a class of rules, which embed predicates for aligning entities across relations and knowledge graphs 𝐺 , assessing correlation between attributes, and extracting data from 𝐺. It unifies logic deduction, correlation models, and data extraction by chasing the data with the rules. We train machine learning models to assess attribute correlation and predict missing values. We develop algorithms for the tuple splitting scheme. 

For more details, see our paper: 

Wenfei Fan, Ziyan Han, Weilong Ren, Ding Wang, Yaoshu Wang, Min Xie, and Mengyi Yan. [*Splitting Tuples of Mismatched Entities*](https://philo-vanguard.github.io/files/papers/Tuple-Splitting-SIGMOD24.pdf). In SIGMOD (2024). ACM.

## Install Required Packages
```
pip install -r requirements.txt
```

## Prepare Datasets
The datasets, dicts, Mc model and Md model used in this project can be downloaded from [This Link](https://drive.google.com/drive/folders/1-Bc20q3hc26cqW-7zJ3R0xHm-t00CrIu?usp=sharing). 
Before running codes, you need to put them in the following directory:
```
mkdir -p /tmp/tuple_splitting/
mv datasets /tmp/tuple_splitting/
mv dict_model /tmp/tuple_splitting/
```

## How To Run
To reproduce the experimental results in the paper, you may run all `exp_*.sh` scripts in `shell/`, such as:
```
cd shell/
./exp_college_overall.sh
```
The results will be output to `/tmp/tuple_splitting/results/`

### Instructions
- `shell/exp_*_overall.sh` are for evaluating overall accuracy (DS, AA and MI), where each phase is based on the output results of the previous phase.  
- `shell/exp_*_separate.sh` are for evaluating separate accuracy (DS, AA and MI), where each phase is based on the groundtruth of the previous phase.  
- `shell/exp_*_time.sh` are for evaluating running time.

### Explanations for buttons in scripts
- **dataID**: the data ID in experiment. {persons: 0, imdb: 1, dblp: 2, college: 3}
- **expOption**: the experimental options, i.e., varyGT, varyKG, varyREE and varyTuples
- **method**: the method name, i.e., SET, SET_noML, SET_noHER and SET_NC
- **useREE**: whether use logic rules (for DS, AA, and MI) in tuple splitting framework
- **useMc**: whether use Mc rules or Mc model (for DS and AA) in tuple splitting framework
- **useKG**: whether use knowledge graph, i.e., HER (for DS and MI) in tuple splitting framework
- **useMd**: whether use Md model (for MI) in tuple splitting framework
- **cuda**: gpu id for using Mc and Md model
- **parallel**: whether use parallel_apply when using Mc(and Md) models to predict scores(and impute), fixed to be false
- **varyGT**: whether change the size of groundtruth in tuple splitting framework (vary Gamma)
- **varyKG**:whether change the size of knowledge graph in tuple splitting framework (vary G)
- **varyREE**: whether change the size of logic rules in tuple splitting framework (vary Sigma)
- **varyTuples**: whether change the size of merged tuples in tuple splitting framework (vary D_T)
- **max_chase_round**: the max round of chase in tuple splitting framework, fixed to be 10
- **Mc_type**: the Mc model type, fixed to be "graph"
- **Mc_conf**: the confidence of Mc model, when we use Mc model to predict scores
- **impute_multi_values**: whether batch impute when using HER in MI. False for dblp and college; True for persons and imdb
- **use_Mc_rules**: whether use Mc rules when using Mc. If not, use Mc model. Fixed to be false.
- **run_DecideTS**: whether run DS, fixed to be true
- **run_Splitting**: whether run AA, fixed to be true
- **run_Imputation**: whether run MI, fixed to be true
- **evaluateDS_syn**: false if use FP of ER method as negative data in tuples_check; true if use tuples_check. Fixed to be false.
- **useDittoRules**: whether use ditto rule in DS, fixed to be true
- **thr_ditto_rule**: the threshold for ditto rule
- **default_training_ratio**: the default training ratio for Mc model
- **default_KG_ratio**: the default ratio for KG size
- **default_ratio**: the default ratio for the rest parameter, i.e., REEs and Gamma
- **evaluate_separately**: whether separately evaluate DS, AA, and MI. If so, use the groundtruth DS results for AA phase; and use the groundtruth AA results for MI phase. True for separate evaluation and False for overall evaluation
- **exist_AA**: whether output accuracy of AA. False when evaluate overall accuracy.
- **output_results**: whether output the intermedia and final results
