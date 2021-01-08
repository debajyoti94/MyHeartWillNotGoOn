''' In this code we will apply the feature engineering techniques
that were applied to train set in the notebook'''

from config import titanic_train_set, titanic_test_set,\
                    output_feature, features_to_drop,\
                    file_delimiter

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import abc
import pickle
import seaborn as sns

class MustHaveForFeatureEngineering(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load_pickled_file(self, filename):
        return

    @abc.abstractmethod
    def dump_file(self, data,
                  filename, path):
        return

    @abc.abstractmethod
    def plot_null_values(self, data):
        # to check if any null values
        # are present in the dataset
        return

class FeatureEngineering(MustHaveForFeatureEngineering):

    # this class is responsible for
    # feature enginering on the  dataset
    def __init__(self):
        self.titanic_train_set = titanic_train_set
        self.titanic_test_set = titanic_test_set

    def fill_age(self, data):
        return

    def label_encoding(self, features_to_encode):
        le = LabelEncoder()
        return

    def cleaning_data(self):

    def load_pickled_file(self, filename):
        return

    def dump_file(self, data,
                  filename, path):
        return

    def plot_null_values(self, data):
        return