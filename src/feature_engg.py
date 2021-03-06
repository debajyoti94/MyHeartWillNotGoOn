""" In this code we will apply the feature engineering techniques
that were applied to train set in the notebook"""

from config import (
    titanic_train_set,
    titanic_test_set,
    features_to_drop,
    male_mean_age,
    female_mean_age,
    null_check_heatmap_file,
)

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import abc
import pickle
import seaborn as sns
from create_folds import SKFold


# creating a template of functions that i absolutely need
# so that i don't forget about it
class MustHaveForFeatureEngineering(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def cleaning_data(self, data):
        return

    @abc.abstractmethod
    def plot_null_values(self, data):
        # to check if any null values
        # are present in the dataset
        return


# inheriting from the abstract class
class FeatureEngineering(MustHaveForFeatureEngineering):

    # this class is responsible for
    # feature enginering on the  dataset
    def __init__(self):
        self.titanic_train_set = titanic_train_set
        self.titanic_test_set = titanic_test_set

    def fill_age(self, data):
        """
        Impute the missing values
        :param data: tuple consisting of age,sex
        :return: mean age, if input age is null
        """
        # PASS A TUPLE, OF (AGE,SEX)
        age = data[0]
        sex = data[1]

        if pd.isnull(age):
            if sex == "male":
                return male_mean_age
            else:
                return female_mean_age
        else:
            return age

    def label_encoding(self, data, features_to_encode):
        """
        Using this function to encode categorical features
         (nominal to be precise)
        :param data: input data, whose feature you want to encode
        :param features_to_encode:
        :return: encoded features
        """
        le = LabelEncoder()
        encoded_feature = le.fit_transform(data[features_to_encode])

        return encoded_feature

    def cleaning_data(self, input_data, dataset_type):
        """
        All the data cleaning and feature engineering
         will happen here. This will work for both train and test data
        :param input_data:
        :param dataset_type:
        :return:
        """

        # drop the features that we do no want to keep
        # while training the model
        cleaned_input_data = input_data.drop(features_to_drop,
                                            axis=1, inplace=False)

        # this part is only for training data
        if dataset_type == "TRAIN":
            # missing age-imputation
            cleaned_input_data["Age"] = cleaned_input_data[["Age",
                                                            "Sex"]].apply(
                                                                        self.fill_age,
                                                                        axis=1
                                                                    )
            # label encoding feature: Sex
            sex_labels = self.label_encoding(cleaned_input_data, "Sex")
            cleaned_input_data["Sex_encoded"] = sex_labels
            cleaned_input_data.drop("Sex", axis=1, inplace=True)
            self.plot_null_values(cleaned_input_data)

            # create the heatmap plot of null values as a check
            self.plot_null_values(cleaned_input_data)

            # create a k-fold column which will be used for cross validation
            cleaned_input_data["kfold"] = -1

            # create stratified kfold cross validation
            skfold_obj = SKFold()
            cleaned_input_data = skfold_obj.create_folds(cleaned_input_data)

        elif dataset_type == "TEST":
            # missing age-imputation
            cleaned_input_data["Age"] = cleaned_input_data[["Age",
                                                            "Sex"]].apply(
                                                                    self.fill_age,
                                                                    axis=1
                                                                )
            # fare column has some missing values
            # so we will fill it up based on the mean value
            cleaned_input_data["Fare"] = cleaned_input_data[["Sex",
                                                             "Fare"]].apply(
                                                                    self.fill_fare,
                                                                    axis=1
                                                                )

            sex_labels = self.label_encoding(cleaned_input_data, "Sex")
            cleaned_input_data["Sex_encoded"] = sex_labels
            cleaned_input_data.drop("Sex", axis=1, inplace=True)
            self.plot_null_values(cleaned_input_data)

        return cleaned_input_data

    def load_pickled_file(self, filename):
        """

        :param filename: file that you want to load
        :return: unpickled file
        """
        with open(filename, "rb") as pickle_handle:
            return pickle.load(pickle_handle)

    def fill_fare(self, gender_fare_tuple):
        """
        Fare feature is missing, so we fill it up based
         on mean values obtained
        :param fare_gender_tuple: (sex,fare)
        :return: imputed fare
        """
        sex = gender_fare_tuple[0]
        fare = gender_fare_tuple[1]

        if pd.isnull(fare):
            if sex == "Male":
                return 8
            else:
                return 18
        else:
            return fare

    def dump_file(self, data, filename, path):
        """

        :param data:  data that we want to dump/serialize
        :param filename: filename that we want to associate to the data
        :param path: where we want to store the data
        :return: 0 if it works out well, -1 if in case something fails
        """
        try:
            with open(str(path + filename), "wb") as pickle_handle:
                pickle.dump(data, pickle_handle)
            return 0

        except Exception:
            return -1

    def plot_null_values(self, data):
        """
        Here we will make a heatmap plot to see if there are any null values
        :param data: input data
        :return: a heatmap, stored on disk
        """

        sns_heatmap_plot = sns.heatmap(data.isnull(),
                                       cmap="Blues", yticklabels=False)
        sns_heatmap_plot.figure.savefig(null_check_heatmap_file)
