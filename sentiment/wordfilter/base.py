from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess


class BaseWordFilter:
    def __init__(self, documents: List[str], labels: List, stopwords=None, **vocab_options):
        self._filtered_words = []
        self._labels = labels
        self.__generate_vocab(documents, **vocab_options)
        self.__vectorize_documents(documents, stopwords)

    def __generate_vocab(self, docs, vocab_size=2000, no_below=100, no_above=0.9):
        doc_tokens = [simple_preprocess(d) for d in docs]
        self._words = Dictionary(doc_tokens)
        self.words.filter_extremes(no_below=no_below, no_above=no_above, keep_n=vocab_size)
        self._words.compactify()

    def __vectorize_documents(self, docs, stopwords):
        vocab = {w: i for i, w in enumerate(self._words.values())}
        vectorizer = CountVectorizer(stop_words=stopwords, vocabulary=vocab)
        self._doc_vecs = vectorizer.fit_transform(docs)

    def fit(self):
        pass

    def save_filter(self, file='models/filter.txt'):
        with open(file, 'wt') as f:
            for word in self.filtered_words:
                f.write('%s\n' % word)
            f.close()

    @property
    def words(self):
        return self._words

    @property
    def filtered_words(self):
        return self._filtered_words

    @property
    def doc_vecs(self):
        return self._doc_vecs

    @property
    def labels(self):
        return self._labels
