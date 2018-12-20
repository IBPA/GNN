### Data Preparation

#### Stratified Sampling
Run ```prep/run_stratify.py directory_name```. This will generate stratified datasets with different sizes (n=10, 20, ..., 100). For each size, it will generate train and test in 5-folds. The files will be saved under ```directory_name/folds``` with names such as ```n10_f2.txt``` for size=10 and fold_id=2. 
