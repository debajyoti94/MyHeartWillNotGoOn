''' In this code we will create K-folds of data for cross validation purpose
In CV: we split the train set into train and validation set
The validation set is an independent set which is used training parameters'''

import pandas as pd
import config
import pickle
from sklearn import model_selection

class KFold:

    def __init__(self):
        '''
        constructor to initialize num of folds
        :param num_folds: num of folds you want to
                            create in the training set
        '''
        self.num_folds = config.num_folds

    def create_folds(self, dataset_df):
        '''

        :param dataset_df:
        :return:
        '''
        # initiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits=self.num_folds)

        # fetch target class
        y = dataset_df['Survived'].values

        # fill in the new kfold column
        for fold_value, (t_,y_index) in enumerate(kf.split(X=dataset_df, y=y)):
            dataset_df.loc[y_index,'kfold'] = fold_value

        return dataset_df



