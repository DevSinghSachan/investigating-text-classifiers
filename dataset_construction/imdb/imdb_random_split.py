# Code to be run in Python 3
# This script creates training, test dataset based on random split.

import numpy as np
from sklearn.model_selection import train_test_split
from nltk import TreebankWordTokenizer


tokenizer = TreebankWordTokenizer()

imdb_train = "/home/dsachan/src/sequence_classification/dataset_construction/imdb_data_train.txt"
imdb_test = "/home/dsachan/src/sequence_classification/dataset_construction/imdb_data_test.txt"

# Train Data path
train_text_path = "/home/dsachan/imdb/train.text"
train_label_path = "/home/dsachan/imdb/train.label"

# Test Data path
test_text_path = "/home/dsachan/imdb/test.text"
test_label_path = "/home/dsachan/imdb/test.label"

num_classes = 5
fp_train = open(imdb_train, 'w')
fp_test = open(imdb_test, 'w')

# Read the arxiv classes
id2class = ['most_negative', 'negative', 'neutral', 'positive', 'most_positive']

# Ratio of train size / total size to keep the ratio similar to lexicon split case
split_ratio = {'most_negative': 1/1.654,
               'negative': 1/1.919,
               'neutral': 1/2.248,
               'positive': 1/1.619,
               'most_positive': 1/1.604}


def read_data(loc_):
    data = list()
    with open(loc_) as fp:
        for line in fp:
            t = line.strip()
            data.append(t)
    return data


train_text = read_data(train_text_path)
y_train = np.array(list(map(int, read_data(train_label_path))))

test_text = read_data(test_text_path)
y_test = np.array(list(map(int, read_data(test_label_path))))

y_train = np.array(list(map(lambda x: np.ceil(x/2.), y_train)))
y_test = np.array(list(map(lambda x: np.ceil(x/2.), y_test)))

all_y = np.concatenate([y_train, y_test])
all_text = train_text + test_text


for i in range(0, num_classes):
    data_list = []

    class_name = id2class[i]
    for x, y in zip(all_text, all_y):
        if y-1 == i:
            data_list.append(x)

    train_set, test_set = train_test_split(data_list, train_size=split_ratio[class_name])

    for text in train_set:
        t = tokenizer.tokenize(text, return_str=True)
        t = t.strip().replace('\t', ' ')
        fp_train.write(str(i) + '\t' + t + '\n')

    for text in test_set:
        t = tokenizer.tokenize(text, return_str=True)
        t = t.strip().replace('\t', ' ')
        fp_test.write(str(i) + '\t' + t + '\n')

fp_train.close()
fp_test.close()