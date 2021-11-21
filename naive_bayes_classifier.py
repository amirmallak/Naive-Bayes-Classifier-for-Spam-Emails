import pandas as pd

from typing import List, Dict

from creating_vocabulary import _vocabulary_creating
from email_classification import classify_email
from pre_processing import _pre_processing
from test_classification import classify_single_email
from word_probabilities import _word_probabilities_calculation


def naive_bayes_classifier() -> None:
    training_set: pd.DataFrame = pd.read_csv(r'data_train.csv')
    test_set: pd.DataFrame = pd.read_csv(r'data_test.csv')
    print('\n--- Training Data ---\n')
    _pre_processing(training_set)
    print('--- Test Data ---\n')
    _pre_processing(test_set)

    print('\nTraining the Naive Bayes Classifier...\n')

    # Creating a vocabulary
    vocabulary: List = _vocabulary_creating(training_set)
    print(f'Our vocabulary\'s length is: {len(vocabulary)}\n')

    # Calculating words probabilities
    non_spam_word_probabilities: Dict = _word_probabilities_calculation(training_set, 0, vocabulary)
    spam_word_probabilities: Dict = _word_probabilities_calculation(training_set, 1, vocabulary)

    # Probability of Spam and Non_Spam emails
    non_spam_probability = sum(1 * (training_set['label'] == 0)) / len(training_set['label'])
    spam_probability = sum(1 * (training_set['label'] == 1)) / len(training_set['label'])

    print('Done Training!\n\n')
    # Testing the Classifier
    print('--- Testing The Classifier ---\n')
    test_non_spam_email = 'Hi Jem, have you finished you\'re assignment yet?'
    test_spam_email = 'Congratulations! You\'re our Winner. Enter you\'re bank account to get the money award'
    print(f'The First Sentence is:')
    print(f'{test_non_spam_email}')
    print('Prediction:')
    classify_single_email(test_non_spam_email,
                          non_spam_probability,
                          spam_probability,
                          non_spam_word_probabilities,
                          spam_word_probabilities)
    print(f'The Second Sentence is:')
    print(f'{test_spam_email}')
    print('Prediction:')
    classify_single_email(test_spam_email,
                          non_spam_probability,
                          spam_probability,
                          non_spam_word_probabilities,
                          spam_word_probabilities)

    # Making Predictions on Test data set
    print('--- Making Predictions on Test Dataset ---\n')
    test_set_prediction = []
    for email in test_set['email']:
        prediction = classify_email(email,
                                    non_spam_probability,
                                    spam_probability,
                                    non_spam_word_probabilities,
                                    spam_word_probabilities)
        test_set_prediction.append(prediction)

    test_set_prediction_data_frame = pd.DataFrame({'Prediction': test_set_prediction})
    test_set_with_prediction = pd.concat([test_set, test_set_prediction_data_frame], axis=1)
    print('The Head of our classifier predictions is:')
    print(f'{test_set_with_prediction.head()}\n')

    # Calculating Accuracy
    print('--- Calculating Classifier\'s Accuracy ---\n')
    correct_classification = 0
    total_data = test_set_with_prediction.shape[0]
    for row in test_set_with_prediction.iterrows():
        row = row[1]
        if row['label'] == row['Prediction']:
            correct_classification += 1

    print(f'Number of Correct Classification Emails is: {correct_classification}')
    print(f'Number of Incorrect Classification Emails is: {total_data - correct_classification}')
    print(f'Naive Bayes Classifier Accuracy: {(correct_classification / total_data) * 100:.2f}%')
