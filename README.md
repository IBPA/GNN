### What is GNN?
Genetic Neural Network (GNN) is an artificial neural network for predicting genome-wide gene expression given gene knockouts and master regulator perturbations. In its core, the GNN maps existing gene regulatory information in its architecture and it uses cell nodes that have been specifically designed to capture the dependencies and non-linear dynamics that exist in gene networks. These two key features make the GNN architecture capable to capture complex relationships without the need of large training datasets.

### Dependencies
+ **Operating Systems**: This code base has been verified on *Ubuntu* 18.04, and *MacOS Sierra* 10.12.6
+ **Programming Languages**: *python* (version >= 3.4) and *lua* (version = 5.1)
+ **Libraries**: *GUROBI* (version >= 6.5), *Torch7*, *Keras*, *TensorFlow* and *pandas*

### Installation
Follow installation steps for details.

### Running
* Step1: prepare a directory containing your input files (with exact names):
	* ``` net.dep ```
	* ``` ge_range.csv ```
	* ``` data_KO.tsv ```
	* ``` data_NonMR.tsv ```
	* ``` data_MR.tsv ```
* Step2: identify filename containing sample ids for training and test sets (call we call this train_test_filename). For stratified sampling and 5-fold cross-validation data, run ```prep/run_stratify.py directory_name```. This will generate stratified datasets with different sizes (n=10, 20, ..., 100). For each size, it will generate train and test in 5-folds. The files will be saved under ```directory_name/folds``` with names such as ```n10_f2.txt``` for size=10 and fold_id=2. The first row contains the indexes for training set and second row corresponds to the test set.
* Step3: change to ```directory_name``` and run ``` th train_test_filename```

Predictions will be saved under ```grnn_pred_train_test_filename.csv```

### Support
For any questions contact Ameen Eetemadi (eetemadi@ucdavis.edu).

### Citation
Eetemadi A and Tagkopoulos I. Genetic Neural Networks: An artificial neural network architecture for capturing gene expression relationships. **Bioinformatics**. 2018. [\[link\]](https://doi.org/10.1093/bioinformatics/bty945)

### Licence
See the [LICENSE](./LICENSE) file for license rights and limitations (Apache2.0).

### Acknowledgement
This work was supported by grants from National Science Foundation (1516695, 1743101 and 1254205).






