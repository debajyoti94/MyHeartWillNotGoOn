# Who will survive on Titanic?

Building a predictive model to see who will survive on titanic ship.
Dataset is taken from [here](https://www.kaggle.com/c/titanic-dataset/data).

The repository has multiple directories, with each serving a different purpose:
- input/: contains the dataset, split into train and test files.
- model/: consists of baseline logistic regression models. Stratified Kfold validation was applied while training the model, with number of folds=5, hence you see 5 models. 
- notebooks/: Consists of one jupyter notebook. It was used for EDA purpose and also experiment with some functions used for feature engineering.
- src/: this directory consists of the source code for the project.
    - config.py: consists of variables which are used all across the code.
    - create_folds.py: used for implementing stratified kfold cross validation.
    - feature_engg.py: used for cleaning the data (test & train) and apply data imputation for missing values in the features.
    - test_functionalities: using pytest module, i define some data sanity checks on the training data.
    - train.py: this file contains the code for implementing the model. The train and the inference stage.

## To train the model use the following command:
  ```python train.py --train skfold```
  
## For inference stage, use:
  ```python train.py --test inference```

