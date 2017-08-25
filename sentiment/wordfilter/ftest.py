from sklearn.feature_selection.univariate_selection import SelectKBest, SelectPercentile

import numpy as np

from sentiment.wordfilter.base import BaseWordFilter


class UnivariateFilter(BaseWordFilter):
    def __init__(self, documents, labels, stopwords, **vocab_options):
        BaseWordFilter.__init__(self, documents, labels, stopwords, **vocab_options)

    def fit(self, k=100, percent=None):
        selector = SelectKBest(k=k)
        selector.fit(self.doc_vecs.todense(), np.asarray(self.labels))

        scores = selector.scores_
        indices = np.argsort(scores)

        if k is not None:
            select = k
        elif percent is not None:
            select = int(len(scores) * percent)
        else:
            raise ValueError('One of `k` or `percent` parameter must be not None.')

        indices = indices[:select]
        self._filtered_words = [self.words[i] for i in indices]
