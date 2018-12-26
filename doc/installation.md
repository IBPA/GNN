### GNN Installation
To install GNN, simply clone this repository:
```git clone https://github.com/IBPA/GNN.git```

### Installing Dependencies
To run GNN successfully, ensure following dependencies are installed.

* **GUROBI:** GNN uses [GUROBI](http://www.gurobi.com/) to solve linear programming problem during training. Download and install version 6.5 or higher following [GUROBI installation instructions](http://www.gurobi.com/downloads/download-center). You may obtain a free academic license if qualified.
* **Torch:** GNN is based on th artificial neural network framework provided by Torch which uses lua language. Follow the [Torch installation instructions](http://torch.ch/docs/getting-started.html#_) (CPU only). We consider moving to [python](https://pytorch.org/) based frameworks such as pytorch in future.
* **Lua packages**: Several lua packages are needed, hence:
	*  ```luarocks install csv```
	*  ```luarocks install cephes ```
	* torch-autograd:
		* ``` git clone https://github.com/ameenetemady/torch-autograd```
		* ```cd torch-autograd``` 
		* ```luarocks make autograd-scm-1.rockspec```
	* gurobi-torch:
		* ```git clone https://github.com/bamos/gurobi.torch```
		* ```cd gurobi.torch```
		* ```luarocks make gurobi-scm-1.rockspec```
* **Python3:** We use python for various scripts including data processing and competing ANN methods. Install python version 3.4 or higher from [here](https://www.python.org/downloads/). The following python modules are needed, which you may intall using pip3:
	* ```pip3 install pandas```
	* ```pip3 install numpy```
	* ```pip3 install keras```
	* ```pip3 install tensorflow```
* **R:** We use R (version 3.4 or higher) for running GENIE3 and visualization of results. Follow [instructions here](https://www.r-project.org/) to install R if don't already have it. To install necessary R packages run the following using R:
	* ```install.packages(c("ggplot2","stringr", "dplyr", "tidyverse", "reshape2", "BiocManager", "argparse"))```
	* ```BiocManager::install("GENIE3")```