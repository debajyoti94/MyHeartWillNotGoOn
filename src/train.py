''' In this code all the modules that we have created will come together
1. Obtain cleaned data using feature engineering module
    a. This module internally calls create folds to obtain stratified kfold splits
    b. Also gets the data cleaned and ready
2. Call models.py and invoke the logistic regression module
    a. To also introduce gridsearch
3. Create the valid and train split then train the model
'''

import config
from feature_engg import FeatureEngineering
from sklearn import linear_model
from sklearn import metrics
import pickle
import pandas as pd
import argparse

# this function is dedicated to putting things together
def run(dataset, fold):
    '''
    This is where we train the model using the dataset
    :param dataset:
    :param fold:
    :return:
    '''

    # creating folds for valid and train
    dataset_train = dataset[dataset.kfold != fold].reset_index(drop=True)
    dataset_valid = dataset[dataset.kfold == fold].reset_index(drop=True)

    # dropping the output feature and kfold feature from X_train
    X_train = dataset_train.drop([config.output_feature, 'kfold'],
                                 axis=1).values

    # keeping only target feature for y_train
    y_train =dataset_train[config.output_feature].values

    # used as test set during training
    X_valid = dataset_valid.drop([config.output_feature, 'kfold'],
                                 axis=1).values
    y_valid = dataset_valid[config.output_feature].values

    # logistic regression model with l2 regulaizer and considering
    # all classes as balanced
    lr = linear_model.LogisticRegression(penalty='l2',
                                         class_weight='balanced')

    # fit the model
    lr.fit(X_train, y_train)

    # obtain predictions from test set
    preds = lr.predict(X_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    p,r,f1_score,support = metrics.precision_recall_fscore_support(y_valid,preds)
    print("---Fold={}---\nAccuracy={}\nPrecision={}\nRecall={}\nF1={}".
          format(fold,accuracy,p,r,f1_score))

    with open(config.baseline_model+'_'+str(fold), 'wb') as pickle_handle:
        pickle.dump(lr,pickle_handle)



if __name__ == "__main__":

    # first we gather the data
    dataset = pd.read_csv(config.titanic_train_set,
                          delimiter=config.file_delimiter)

    # create object of FeatureEngineering class
    fr_obj = FeatureEngineering()
    cleaned_data = fr_obj.cleaning_data(dataset,dataset_type="TRAIN")

    # next we inititalize the arguments using argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str,
                        help="This parameter will simply train the model")
    parser.add_argument("--test", type=None,
                        help="This parameter will use the test set and derive metrics")

    args = parser.parse_args()

    # then we call the run() function
    if args.train == 'skfold':
        run(cleaned_data, 0)
        run(cleaned_data, 1)
        run(cleaned_data, 2)
        run(cleaned_data, 3)
        run(cleaned_data, 4)
    if args.test:
        print("Code not ready yet for model inference.")