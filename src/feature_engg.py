''' In this code we will apply the feature engineering techniques
that were applied to train set in the notebook'''

from config import titanic_train_set, titanic_test_set,\
                    output_feature, features_to_drop,\
                    file_delimiter, male_mean_age,\
                    female_mean_age

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import abc
import pickle
import seaborn as sns

# creating a template of functions that i absolutely need
# so that i don't forget about it
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

# inheriting from the abstract class
class FeatureEngineering(MustHaveForFeatureEngineering):

    # this class is responsible for
    # feature enginering on the  dataset
    def __init__(self):
        self.titanic_train_set = titanic_train_set
        self.titanic_test_set = titanic_test_set

    def fill_age(self, data):
        '''
        Impute the missing values
        :param data: tuple consisting of age,sex
        :return: mean age, if input age is null
        '''
        # PASS A TUPLE, OF (AGE,SEX)
        age = data[0]
        sex = data[1]

        if pd.isnull(age):
            if sex is 'male':
                return male_mean_age
            else:
                return female_mean_age
        else:
            return age


    def label_encoding(self, data,
                       features_to_encode):
        '''
        Using this function to encode categorical features
         (nominal to be precise)
        :param data: input data, whose feature you want to encode
        :param features_to_encode:
        :return: encoded features
        '''
        le = LabelEncoder()
        encoded_feature = le.fit_transform(data[features_to_encode])

        return encoded_feature

    def cleaning_data(self):

    def load_pickled_file(self, filename):
        '''

        :param filename: file that you want to load
        :return: unpickled file
        '''
        with open(filename, 'rb') as pickle_handle:
            return pickle.load(pickle_handle)

    def dump_file(self, data,
                  filename, path):
        '''

        :param data:  data that we want to dump/serialize
        :param filename: filename that we want to associate to the data
        :param path: where we want to store the data
        :return: 0 if it works out well, -1 if in case something fails
        '''
        try:
            with open(str(path+filename), 'wb') as pickle_handle:
                pickle.dump(data, pickle_handle)
            return 0

        except Exception:
            return -1

    def plot_null_values(self, data):
        '''
        Here we will make a heatmap plot to see if there are any null values
        :param data: input data
        :return: a heatmap, stored on disk
        '''

        return