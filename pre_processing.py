import numpy as np
import pandas as pd


def _pre_processing(data_set: pd.DataFrame) -> None:
    # Dropping NAN Rows
    data_set.dropna(axis=0, how='all', inplace=True)
    data_set.reset_index(drop=True, inplace=True)
    print(f'Shape of data file is: {data_set.shape}\n')
    print('Data\'s Header is: ')
    print(f'{data_set.head()}\n')
    non_spam_training_data = np.sum(data_set['label'] == 0)
    spam_data = np.sum(data_set['label'] == 1)
    total_data = data_set['label'].size
    non_spam_percentage = non_spam_training_data / total_data
    spam_percentage = 1 - non_spam_percentage
    print(f'Number of non spam data: {non_spam_training_data}, percentage: {non_spam_percentage:.2f}%')
    print(f'Number of spam data: {spam_data}, percentage: {spam_percentage:.2f}%\n')
    # Data Cleaning
    data_set['email'].replace(r'\W', ' ', regex=True, inplace=True)
    data_set['email'].replace('_', ' ', regex=True, inplace=True)
    data_set['email'] = data_set['email'].str.lower()

    # Handling 'nan'
    data_set['email'].fillna(' ', inplace=True)
