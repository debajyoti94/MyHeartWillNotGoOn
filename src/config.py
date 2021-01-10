""" In config.py we can setup the hyper-parameters
 of the model and of the script in general
This file will contain the variables which are useful
 for training the model."""

titanic_train_set = "../input/train.csv"
titanic_test_set = "../input/test.csv"

# for feature engineering
output_feature = "Survived"
features_to_drop = ["Cabin", "PassengerId", "Embarked", "Ticket", "Name"]
male_mean_age = 29
female_mean_age = 25

# for saving the cleaned data
file_delimiter = ","
baseline_X_train_set = "../input/baseline_X_train_set.pickle"
baseline_Y_train_set = "../input/baseline_Y_train_set.pickle"

baseline_X_test_set = "../input/baseline_X_test_set.pickle"
baseline_Y_test_set = "../input/baseline_Y_test_set.pickle"


# for saving the model
baseline_model = "../models/LR_Baseline"
best_baseline_model = "../models/LR_Baseline_2"
fe_model = "../models/LR_FE"

# plots
null_check_heatmap_file = "../plots/heatmap_null_features_check.png"

# number of folds for cross validation
num_folds = 5
