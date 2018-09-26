#!/bin/bash

# 0) Under data/dream5_ecoli/expr1/
# 1) for i=1,10: divide data (dream5_ecoli/input) into e[i]/d_net and e[i]/d_eval
# 3) for i=1,10: sbatch ./prep/run_GENIE3.R e[i]/d_net/
# 4) for i=1,10: ./prep/run_generate_module_data.py e[i]/d_net/edges_inferred.tsv e[i]/d_eval/data_all_unique.tsv e[i]/modules/
# 5) ./prep/run_aggregate_multiple_modules.py 