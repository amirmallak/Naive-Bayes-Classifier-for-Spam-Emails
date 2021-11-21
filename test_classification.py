import re
import numpy as np

from typing import Dict


def classify_single_email(email: str, p_non_spam: float, p_spam: float, p_word_non_spam: Dict, p_word_spam: Dict) \
                          -> None:
    email = re.sub(r'\W', ' ', email)
    email = email.lower().split()

    # p_non_spam_given_email = p_non_spam
    # p_spam_given_email = p_spam
    p_non_spam_given_email = np.log(p_non_spam)
    p_spam_given_email = np.log(p_spam)
    for word in email:
        if word in p_word_non_spam:
            # p_non_spam_given_email *= p_word_non_spam[word]
            p_non_spam_given_email += np.log(p_word_non_spam[word])
        if word in p_word_spam:
            # p_spam_given_email *= p_word_spam[word]
            p_spam_given_email += np.log(p_word_spam[word])
        # if p_non_spam_given_email < 1e-100 or p_spam_given_email < 1e-100:
        #     p_non_spam_given_email *= 1e100
        #     p_spam_given_email *= 1e100

    print(f'P(Non_Spam|email) = {np.power(10, p_non_spam_given_email)}')
    print(f'P(Spam|email) = {np.power(10, p_spam_given_email)}')

    if p_non_spam_given_email > p_spam_given_email:
        print('Label: Non Spam\n')
    elif p_non_spam_given_email < p_spam_given_email:
        print('Label: Spam\n')
    else:
        print('Can\'t make a decision deu to equal probabilities!\n')
