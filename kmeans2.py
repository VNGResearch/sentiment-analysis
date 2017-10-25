from argparse import ArgumentParser
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import collections
import numpy as np
import sys
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

os.environ['KERAS_BACKEND'] = 'theano'
sys.path.append('./socialsent3')

from socialsent3.polarity_induction_methods import random_walk
from socialsent3.representations.representation_factory import create_representation
from socialsent3.util import dict2csv
from socialsent3 import seeds


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--train', default='./data/imdb/train.tsv', dest='TRAIN',
                        help='File containing training data')
    parser.add_argument('--test', default='./data/imdb/val.tsv', dest='TEST',
                        help='File containing testing data')
    parser.add_argument('--embed', default='./data/embeddings/imdb.en.vec', dest='EMBED',
                        help='File containing word embedding')
    parser.add_argument('--cutoff', default=40, type=int, dest='CUTOFF',
                        help='Number of most positive and negative words to filter out.')
    parser.add_argument('--lsa', dest='LSA', type=int, default=0,
                        help='LSA reduction size')
    parser.add_argument('--algo', dest='ALGO', default=None)
    return parser.parse_args()


HALF_LEN = 2


def sent2rep(sent, word_list=None, **kwargs):
    words = nltk.word_tokenize(sent)
    # noinspection PyArgumentList
    counter = collections.Counter(words)
    rep = np.zeros(len(word_list))
    for i, word in enumerate(word_list):
        rep[i] = counter.get(word, 0)
        # rep[i] = 1 if word in counter.keys() else 0

    return rep


def evaluate(true_labels, pred_labels):
    print('Homogeneity score: %.3f' % homogeneity_score(true_labels, pred_labels))
    print('Completeness score: %.3f' % completeness_score(true_labels, pred_labels))


def filter_polarities(polarities: dict, cutoff):
    l = polarities.items()
    sorted_l = sorted(l, key=lambda it: it[1])

    neg = sorted_l[:cutoff]
    # neg = []
    pos = sorted_l[-cutoff:]
    newp = dict(neg + pos)

    return newp


def main(args):
    print('Loading data...')
    train_sents, train_labels = [], []
    with open(args.TRAIN, 'rt') as f:
        lines = f.readlines()
        for l in lines:
            cols = l.split('\t')
            train_labels.append(1 if cols[0] == 'Positive' else 0)
            train_sents.append(cols[1].strip())

    test_sents, test_labels = [], []
    with open(args.TEST, 'rt') as f:
        lines = f.readlines()
        for l in lines:
            cols = l.split('\t')
            test_labels.append(1 if cols[0] == 'Positive' else 0)
            test_sents.append(cols[1].strip())

    pos_seeds, neg_seeds = seeds.review_seeds()
    print('Creating word vectors...')
    embeddings = create_representation("FULL", args.EMBED, 100,
                                       limit=50000)
    print('Calculating polarities...')
    polarities = random_walk(embeddings, pos_seeds, neg_seeds, beta=0.99, nn=10,
                             sym=True, arccos=True)

    print('Filtering polarities...')
    polarities = filter_polarities(polarities, args.CUTOFF)

    print('Storing polarities...')
    dict2csv(polarities, path='./data/polarities/filtered.csv')

    word_list = list(polarities.keys())
    train_reps, test_reps = [], []

    if args.ALGO == 'tf-idf':
        tfidf = TfidfVectorizer(vocabulary=word_list, tokenizer=nltk.word_tokenize)

        print('Creating training sentence representations...')
        train_reps = tfidf.fit_transform(train_sents)

        print('Creating testing sentence representations...')
        test_reps = tfidf.fit_transform(test_sents)
    else:
        print('Creating training sentence representations...')
        for i, sent in enumerate(train_sents):
            print('\t%d/%d' % (i + 1, len(train_sents)), end='\r')
            rep = sent2rep(sent, word_list=word_list)
            train_reps.append(rep)
        print()

        print('Creating testing sentence representations...')
        for i, sent in enumerate(test_sents):
            print('\t%d/%d' % (i + 1, len(test_sents)), end='\r')
            rep = sent2rep(sent, word_list=word_list)
            test_reps.append(rep)
        print()

    if args.LSA != 0:
        print('Transforming w/ LSA...')
        svd = TruncatedSVD(args.LSA)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        train_reps = lsa.fit_transform(train_reps)
        test_reps = lsa.fit_transform(test_reps)

    km = KMeans(n_clusters=2, verbose=1, max_iter=10000)
    train_preds = km.fit_predict(train_reps)
    test_preds = km.predict(test_reps)

    print('\nMetrics on train set:')
    evaluate(train_labels, train_preds)
    print('\nMetrics on test set:')
    evaluate(test_labels, test_preds)


if __name__ == '__main__':
    main(parse_arguments())
