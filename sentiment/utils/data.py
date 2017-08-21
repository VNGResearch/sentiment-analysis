import random
import re


def load_data(path, text_transform, label_transform, nlines=None, **kwargs):
    f = open(path, 'rt')
    lines = f.readlines()
    f.close()

    random.shuffle(lines)
    lines = lines[:nlines] if nlines is not None else lines
    labels, texts = [], []
    for line in lines:
        label, text = re.split('[ \n]', line, maxsplit=2)
        labels.append(label)
        texts.append(text)
    return text_transform(texts, **kwargs), label_transform(labels, **kwargs)


