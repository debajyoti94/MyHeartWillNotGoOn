""" In config.py we can setup the hyper-parameters of the model and of the script in general
This file will contain the variables which are useful for training the model."""

titanic_train_set = "../input/train.csv"
titanic_test_set = "../input/test.csv"

# for feature engineering
output_feature = 'Survived'
features_to_drop = ['Cabin', 'PassengerId', 'Embarked',
                    'Ticket', 'Name']

# for saving the cleaned data
baseline_train_set = "../input/baseline_train_set.pickle"
baseline_test_set = "../input/baseline_test_set.pickle"


# for saving the model
baseline_model = "../models/LR_Baseline.pickle"
fe_model = "../models/LR_FE.pickle"
