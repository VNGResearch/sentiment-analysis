from sentiment.wordfilter import UnivariateFilter
from sentiment.utils import load_data


def main(args):
    docs, labels = load_data('data/text_emotion/texts.tsv')
    mfilter = UnivariateFilter(docs, labels, stopwords='english',
                               vocab_size=20000, no_below=10, no_above=0.95)
    mfilter.fit()
    mfilter.save_filter()


if __name__ == '__main__':
    main(None)
