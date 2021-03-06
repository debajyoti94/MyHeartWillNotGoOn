""" In this code we will create K-folds of data for
 cross validation purpose
In CV: we split the train set into train and validation set
The validation set is an independent set which is used
training parameters"""

import config
from sklearn import model_selection


class SKFold:
    def __init__(self):
        """
        constructor to initialize num of folds
        :param num_folds: num of folds you want to
                            create in the training set
        """
        self.num_folds = config.num_folds

    def create_folds(self, dataset_df):
        """

        :param dataset_df:
        :return:
        """
        # initiate the kfold class from model_selection module
        # keeping shuffle and random state so that
        # we can replicate the experiments

        kf = model_selection.StratifiedKFold(
            n_splits=self.num_folds, shuffle=True, random_state=0
        )

        # fetch target class
        y = dataset_df["Survived"].values

        # fill in the new kfold column
        for fold_value, (t_, y_index) in enumerate(
                                        kf.split(X=dataset_df,
                                                 y=y)):
            dataset_df.loc[y_index, "kfold"] = fold_value

        return dataset_df
