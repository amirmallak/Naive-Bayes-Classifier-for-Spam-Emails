import re
import numpy as np

from typing import Optional, Dict


def classify_email(email: str, p_non_spam: float, p_spam: float, p_word_non_spam: Dict, p_word_spam: Dict) \
                   -> Optional[int]:
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
        # if p_non_spam_given_email < 1e-10 or p_spam_given_email < 1e-10:
        #     p_non_spam_given_email *= 1e10
        #     p_spam_given_email *= 1e10

    if p_non_spam_given_email > p_spam_given_email:
        return 0
    elif p_non_spam_given_email < p_spam_given_email:
        return 1
    else:
        return None
