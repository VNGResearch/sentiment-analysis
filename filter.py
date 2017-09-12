from argparse import ArgumentParser

from sentiment.wordfilter import UnivariateFilter
from sentiment.utils import load_data


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data', default='data/amazon/train.tsv', dest='DATA')
    parser.add_argument('--size', default=20000, type=int, dest='SIZE')
    parser.add_argument('--no-below', default=10, type=int, dest='BELOW')
    parser.add_argument('--no-above', default=0.99, type=float, dest='ABOVE')
    parser.add_argument('-k', default=30, type=int, dest='K')

    return parser.parse_args()


def main(args):
    docs, labels = load_data(args.DATA)
    mfilter = UnivariateFilter(docs, labels, stopwords='english',
                               vocab_size=args.SIZE, no_below=args.BELOW, no_above=args.ABOVE)
    mfilter.fit(k=args.K)
    mfilter.save_filter()


if __name__ == '__main__':
    main(parse_arguments())
