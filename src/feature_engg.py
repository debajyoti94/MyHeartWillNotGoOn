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

    def cleaning_data(self, input_data):
        '''
        All the data cleaning and feature engineering
         will happen here. This will work for both train and test data
        :param input_data:
        :return: cleaned_data,X,y in array format
        '''

        # drop the features that we do no want to keep
        # while training the model
        cleaned_input_data = input_data.drop(features_to_drop, axis=1,
                                             inplace=False)

        # missing age-imputation
        cleaned_input_data['Age'] = cleaned_input_data[['Age', 'Sex']].apply(
                                                                        self.fill_age, axis=1)
        # label encoding feature: Sex
        sex_labels = self.label_encoding(cleaned_input_data,'Sex')
        cleaned_input_data['Sex_encoded'] = sex_labels

        # splitting features and output (X,y)
        X_data = cleaned_input_data.drop('Survived', axis=1,
                                         inplace=False).values

        y_data = cleaned_input_data['Survived'].values

        return cleaned_input_data, X_data, y_data

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
        try:
            sns_heatmap_plot = sns.heatmap(data.isnull(), cmap='Blues', yticklabels=False)
            sns_heatmap_plot.savefig('../plots/heatmap_null_features_check.png')

            # if all succeeds
            return 0
        except Exception:
            # if things fail
            return -1
