from argparse import ArgumentParser
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import homogeneity_score, completeness_score

import nltk
import numpy as np
import sys
import os
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
    return parser.parse_args()


HALF_LEN = 2


def sent2rep(sent, polarities):
    words = nltk.word_tokenize(sent)
    rep = np.asarray([polarities.get(w, 0.5) for w in words])

    rep = np.sort(rep)
    rep = np.hstack((rep[:HALF_LEN], rep[-HALF_LEN:]))
    if len(rep) < 2 * HALF_LEN:
        rep = np.hstack((rep, np.ones((2*HALF_LEN - len(rep),)) * 0.5))

    return rep


def evaluate(true_labels, pred_labels):
    print('Homogeneity score: %.3f' % homogeneity_score(true_labels, pred_labels))
    print('Completeness score: %.3f' % completeness_score(true_labels, pred_labels))


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
                                       limit=30000)
    print('Calculating polarities...')
    polarities = random_walk(embeddings, pos_seeds, neg_seeds, beta=0.99, nn=10,
                             sym=True, arccos=True)

    print('Storing polarities...')
    dict2csv(polarities, path='./data/polarities/default.csv')

    print('Creating training sentence representations...')
    train_reps = []
    for i, sent in enumerate(train_sents):
        print('\t%d/%d' % (i+1, len(train_sents)), end='\r')
        rep = sent2rep(sent, polarities)
        train_reps.append(rep)
    print()

    print('Creating testing sentence representations...')
    test_reps = []
    for i, sent in enumerate(test_sents):
        print('\t%d/%d' % (i + 1, len(test_sents)), end='\r')
        rep = sent2rep(sent, polarities)
        test_reps.append(rep)
    print()

    km = KMeans(n_clusters=2, verbose=1, max_iter=10000)
    train_preds = km.fit_predict(train_reps)
    test_preds = km.predict(test_reps)

    print('\nMetrics on train set:')
    evaluate(train_labels, train_preds)
    print('\nMetrics on test set:')
    evaluate(test_labels, test_preds)


if __name__ == '__main__':
    main(parse_arguments())
