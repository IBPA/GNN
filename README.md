### What is GNN?
Genetic Neural Network (GNN) is an artificial neural network for predicting genome-wide gene expression given gene knockouts and master regulator perturbations. In its core, the GNN maps existing gene regulatory information in its architecture and it uses cell nodes that have been specifically designed to capture the dependencies and non-linear dynamics that exist in gene networks. These two key features make the GNN architecture capable to capture complex relationships without the need of large training datasets.

### Dependencies
+ **Operating Systems**: This code base has been verified on *Ubuntu* 18.04, and *MacOS Sierra* 10.12.6
+ **Programming Languages**: *python* (version >= 3.4) and *lua* (version = 5.1)
+ **Libraries**: *GUROBI* (version >= 6.5), *Torch7*, *Keras*, *TensorFlow* and *pandas*

### Installation
Follow [installation steps](doc/installation.md) for details.

### Running
* **Step1**: prepare a directory containing your input files (with exact names):
	* ``` net.dep ```
	* ``` ge_range.csv ```
	* ``` data_KO.tsv ```
	* ``` data_NonMR.tsv ```
	* ``` data_MR.tsv ```
* **Step2**: identify ```train_test_filename``` containing comma separated sample ids for training (first row) and test (second row) sets. To generate using stratified sampling and 5-fold cross-validation follow instructions [here](doc/data_preparation.md#stratified-sampling).
* **Step3**: run ```th trainPred.lua directory_name train_test_filename```

Predictions will be saved as ```grnn_pred_[train_test_filename].csv```

### Support
For any questions contact Ameen Eetemadi (eetemadi@ucdavis.edu).

### Citation
Eetemadi A and Tagkopoulos I. Genetic Neural Networks: An artificial neural network architecture for capturing gene expression relationships. **Bioinformatics**. 2018. [\[link\]](https://doi.org/10.1093/bioinformatics/bty945)

### Licence
See the [LICENSE](./LICENSE) file for license rights and limitations (Apache2.0).

### Acknowledgement
This work was supported by grants from National Science Foundation (1516695, 1743101 and 1254205).






