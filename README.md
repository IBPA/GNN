### What is GNN?
Genetic Neural Network (GNN) is an artificial neural network for predicting genome-wide gene expression given gene knockouts and master regulator perturbations. In its core, the GNN maps existing gene regulatory information in its architecture and it uses cell nodes that have been specifically designed to capture the dependencies and non-linear dynamics that exist in gene networks. These two key features make the GNN architecture capable to capture complex relationships without the need of large training datasets.

### Dependencies
+ **Operating Systems**: Verified on *Ubuntu* 18.04, and *MacOS Sierra* 10.12.6
+ **Programming Languages**: *python* (version >= 3.4) and *lua* (version = 5.1)
+ **Libraries**: *GUROBI* (version >= 6.5), *Torch7*, *Keras*, *TensorFlow* and *pandas*

### Installation
Follow [installation steps](doc/installation.md) for details.

### How to use GNN
**Training.** To train a GNN, you only need the GE dataset. There is also the option to also supply the known/inferred TRN network as a tsv file (see bellow for file formats). If no TRN file is provided, the GNN trainer will first run the GENIE3 method, create an inferred TRN network and it will use that to train the GNN. By default, the trained model will be saved under directory named `model_dir`.To train the GNN, write:

``` ./train.py --dataset dataset.csv --method GNN [--trn net.tsv] [--output-model-dir model_dir]```

**Prediction.** A trained GNN can be used to predict a new gene expression profile:

``` ./predict.py --input gnn_input.csv [--load-model-dir model_dir] --output gnn_pred.csv```

**File formats.**

* ```dataset.csv```([e.g.](./data/hello_world/dataset.csv)): Each row corresponds to GE profile of an experiment. First column contains knockout genes (separated by ```&``` if multiple knockouts). Each other column represents the expression of a gene. First row encodes column names.
* ```GNN```([e.g.](MlinearGNN)): Build and train GNN model.
* ```net.tsv```([e.g.](./data/hello_world/net.tsv)): Each row encodes a single regulatory relationship. First column corresponds to transcription factor (TF) gene and second column to the gene regulated by TF.  
* ```gnn_input.csv```([e.g.](./data/hello_world/gnn_input.csv)): It encodes knockout information (column1) and the expression of master regulator (MR) genes (column2 to last). Each row, corresponds to an experiment. The list of MR genes can be found from ```model_dir/MR_genes.csv```. First row encodes column names.

* ```gnn_pred.csv```([e.g.](./data/hello_world/gnn_pred.csv)): Each row represents predicted gene expressions corresponding to a row of ```gnn_input.csv``` above. First row encodes the gene names.

<!---
### Running
* **Step1**: prepare a directory containing your input files (with exact names):
	* ``` net.dep ```
	* ``` Ranges_GE.csv ```
	* ``` KO_NonMR.tsv ```
	* ``` GE_NonMR.tsv ```
	* ``` GE_MR.tsv ```
* **Step2**: identify ```train_test_filename``` containing comma separated sample ids for training (first row) and test (second row) sets. To generate using stratified sampling and 5-fold cross-validation follow instructions [here](doc/data_preparation.md#stratified-sampling).
* **Step3**: run ```th trainPred.lua directory_name train_test_filename```

Predictions will be saved as ```grnn_pred_[train_test_filename].csv```
-->


### Performance Benchmarks
Follow instructions [here](doc/benchmarks.md) to reproduce performance benchmarks of our [article](https://doi.org/10.1093/bioinformatics/bty945) (Figure 3 and 4).

### Support
For any questions contact Ameen Eetemadi (eetemadi@ucdavis.edu).

### Citation
Eetemadi A and Tagkopoulos I. Genetic Neural Networks: An artificial neural network architecture for capturing gene expression relationships. **Bioinformatics**. 2018. [\[link\]](https://doi.org/10.1093/bioinformatics/bty945)

### Licence
See the [LICENSE](./LICENSE) file for license rights and limitations (Apache2.0).

### Acknowledgement
This work was supported by grants from National Science Foundation (1516695, 1743101 and 1254205).






