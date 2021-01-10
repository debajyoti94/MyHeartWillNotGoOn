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
def train_model(dataset, fold):
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
    # print(X_train.head())
    # print(X_train.shape)

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

def inference_time(model, test_data):
    '''
    This is where we apply obtain performance on the test set
    :param model:
    :param test_data:
    :return:
    '''
    print(test_data.isnull().any())
    X_test = test_data.values

    for input in X_test:

        print(input)
        preds = model.predict(input)
        print(preds)
    # accuracy = metrics.accuracy_score(y_test, preds)
    # p,r,f1_score,support = metrics.precision_recall_fscore_support(y_test,
    #                                                                preds)
    #
    # print("Test set accuracy = {}".format(accuracy))
    # print("---Test set---\nAccuracy={}\nPrecision={}\nRecall={}\nF1={}".
    #       format(accuracy, p, r, f1_score))


if __name__ == "__main__":

    # we inititalize the arguments using argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str,
                        help="This parameter will simply train the model")
    parser.add_argument("--test", type=None,
                        help="This parameter will use the test set and derive metrics")

    args = parser.parse_args()

    # then we call the run() function
    if args.train == 'skfold':
        # first we gather the data
        dataset = pd.read_csv(config.titanic_train_set,
                              delimiter=config.file_delimiter)

        # create object of FeatureEngineering class
        fr_obj = FeatureEngineering()
        cleaned_data = fr_obj.cleaning_data(dataset, dataset_type="TRAIN")

        # train the model
        train_model(cleaned_data, 0)
        train_model(cleaned_data, 1)
        train_model(cleaned_data, 2)
        train_model(cleaned_data, 3)
        train_model(cleaned_data, 4)
    if args.test == 'inference':
        # first we gather the data
        dataset = pd.read_csv(config.titanic_test_set,
                              delimiter=config.file_delimiter)

        # create object of FeatureEngineering class
        fr_obj = FeatureEngineering()
        cleaned_data = fr_obj.cleaning_data(dataset, dataset_type="TEST")

        # load model
        with open(config.best_baseline_model, 'rb') as pickle_handle:
            inference_model = pickle.load(pickle_handle)

        inference_time(inference_model, cleaned_data)

        print("Code not ready yet for model inference.")