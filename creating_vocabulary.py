import pandas as pd

from typing import List


def _vocabulary_creating(data_set: pd.DataFrame) -> List:
    # Creating a List which contains all the words in our data set (in a unique manner)
    data_set['email'] = data_set['email'].str.split()
    dimensions = set()
    for email in data_set['email']:
        for word in email:
            dimensions.add(word)
    dimensions = list(dimensions)

    return dimensions
