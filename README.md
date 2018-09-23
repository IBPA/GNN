# Gene Regulatory Network (GNN)

## Dependencies
* [torch7](http://torch.ch/docs/getting-started.html)
* luarocks install csv
* luarocks install cephes
* git clone: [MyCommon](https://github.com/ameenetemady/MyCommon)
* git clone: [torch-autograd](https://github.com/ameenetemady/torch-autograd)
* cd torch-autograd; luarocks make autograd-scm-1.rockspec
* module load gurobi
* git clone [gurobi.torch](https://github.com/bamos/gurobi.torch)
* cd gurobi.torch; luarocks make gurobi-scm-1.rockspec
* module load gurobi

### Gene Regulatory Network
Architecture of GNN model is informed by a transcription regulatory network (TRN). When the TRN is unknown, network inference methods such as GENIE3 can be used. To run GENIE3 on a gene expression dataset, run: ```./prep/run_GENIE3.R directory_path```

For **input**, ```directory_path ``` should contain files: ```data.tsv```, ```gene_names.tsv```, ```experiments.tsv``` and ```tf_names.tsv```. See ```./data/dream5_r1/``` for file formats. The **output** will be stored in ```trn_edges.tsv``

### Real Data Evaluation
```
./prep/run_divide_data.py ./data/dream5/input/ ./data/dream5/s1/ ./data/dream5/s2/
./prep/run_GENIE3.R ./data/dream5/s1/
./prep/run_generate_module_data.py ./data/dream5/s1/ ./data/dream5/s2/ data/dream5/s2/experiments/
./slurm/run_GNN.sh ./data/dream5/s2/experiments/
./slurm/aggregate_experiment_results.sh ./data/dream5/s2/experiments/
./vis/vis_results.R ./data/dream5/s2/experiments/
```



