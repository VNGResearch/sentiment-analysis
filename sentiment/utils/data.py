import random
import re

__noop = lambda x, **kwargs: x


def load_data(path, text_transform=__noop, label_transform=__noop, nlines=None, **kwargs):
    f = open(path, 'rt')
    lines = f.readlines()
    f.close()

    random.shuffle(lines)
    lines = lines[:nlines] if nlines is not None else lines
    labels, texts = [], []
    for line in lines:
        label, text = re.split('[ \n]', line, maxsplit=1)
        labels.append(label)
        texts.append(text)
    return text_transform(texts, **kwargs), label_transform(labels, **kwargs)
