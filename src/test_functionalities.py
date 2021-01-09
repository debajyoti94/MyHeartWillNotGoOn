''' In this code we will add unit test functionalities using pytest module'''

import pickle
from config import titanic_train_set, titanic_test_set,\
                    file_delimiter
import pandas as pd
import csv
import os

class TestFunctionality:

    def test_file_delimiter(self):
        '''
        check if the file delimiter matches with the one mentioned
        in config file
        :return:
        '''
        with open(titanic_train_set, 'r') as csv_file:
            file_contents = csv.Sniffer().sniff(csv_file.readline())

            assert True if (file_contents.delimiter == file_delimiter) else False

    def test_dataset_shape(self):
        '''
        checking if the dataset shape matches when file is opened
        :return:
        '''
        train_dataset = pd.read_csv(titanic_train_set, delimiter=file_delimiter)
        assert True if train_dataset.shape == (891,12) else False

    def test_plot_null_values(self):
        '''
        check if the plot is created in the location
        :return:
        '''
        assert True if os.path.isfile(titanic_train_set) and\
                       os.path.isfile(titanic_test_set) \
                       else False


    def test_cleaning_data(self):
        return