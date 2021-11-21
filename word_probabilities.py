import pandas as pd

from typing import Optional, List, Dict


def _word_probabilities_calculation(data_set: pd.DataFrame, label: Optional[int], dimensions: List) -> Dict:
    specific_data_set = data_set[data_set['label'] == label]  # Non Spam or Spam data set
    appearances: Dict[List] = {word: [0] * len(specific_data_set['email']) for word in dimensions}
    for index, email in enumerate(specific_data_set['email']):
        for word in email:
            appearances[word][index] += 1

    word_list = []
    count_list = []
    for key in appearances.keys():
        count_list.append(sum(appearances[key]))
        word_list.append(key)

    # Counting the frequencies of word in the training data set
    frequencies = dict(zip(word_list, count_list))

    # Calculating word probabilities
    probabilities = []
    alpha_laplace_smoothing = 0.5
    # alpha_laplace_smoothing = alpha
    total_words_count = specific_data_set['email'].apply(len).sum()
    for word in dimensions:
        word_appearances = frequencies[word]
        word_probability = (word_appearances + alpha_laplace_smoothing) / \
                           (total_words_count + (alpha_laplace_smoothing * len(dimensions)))
        probabilities.append(word_probability)
    word_probabilities = dict(zip(word_list, probabilities))

    return word_probabilities
