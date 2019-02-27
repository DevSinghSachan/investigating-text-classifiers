# Code to be run in Python 3
# This script creates training, test, dev dataset based on generated lexicons

import os
import numpy as np
from collections import defaultdict
from util import create_data_partitions
from nltk import TreebankWordTokenizer

# tokenizer = TreebankWordTokenizer()

min_vocab = 20

lexicons_dir = 'aclImdb_lexicons'     # Lexicons directory

# Train Data path
train_data_path = "/home/dsachan/src/sequence_classification/text_classification/data/aclImdb_tok/train.txt"

# Test Data path
test_data_path = "/home/dsachan/src/sequence_classification/text_classification/data/aclImdb_tok/test.txt"

imdb_train = "/home/dsachan/src/sequence_classification/dataset_construction/train_lexicon.txt"
imdb_test = "/home/dsachan/src/sequence_classification/dataset_construction/test_lexicon.txt"


num_classes = len(os.listdir(lexicons_dir))
fp_train = open(imdb_train, 'w')
fp_test = open(imdb_test, 'w')

# Read the arxiv classes
id2class = ['negative', 'positive']


def read_data(loc_):
    data = list()
    label = list()
    with open(loc_) as fp:
        for line in fp:
            tokens = line.strip().split('\t')
            l, t = tokens[0], ' '.join(tokens[1:])
            data.append(t)
            label.append(l)
    return label, data


print("Reading imdb reviews")
train_label, train_text = read_data(train_data_path)
y_train = np.array(list(map(int, train_label)))

test_label, test_text = read_data(test_data_path)
y_test = np.array(list(map(int, test_label)))

y_train = np.concatenate([y_train, y_test])
train_text = train_text + test_text


# Read the lexicon files
lexicons_dict = defaultdict(list)
for category in os.listdir(lexicons_dir):
    path_ = lexicons_dir + '/' + category
    with open(path_) as fp:
        for line in fp:
            word = line.split('\t')[0]
            lexicons_dict[category].append(word)


# Constructing the Laplacian for graph of every class
for i in range(0, num_classes):
    data_list = []
    edge_list = []

    class_name = id2class[i]
    for x, y in zip(train_text, y_train):
        if y == i:
            data_list.append(x)

    vocabulary = lexicons_dict[class_name]
    num_words = len(vocabulary)
    partition1, partition2, ratio = create_data_partitions(data_list, vocabulary)

    loop_num = 0
    while ratio < 1.0:
        loop_num += 1
        vocabulary = vocabulary[:-200]
        num_words = len(vocabulary)
        partition1, partition2, ratio = create_data_partitions(data_list, vocabulary)
        print(str(loop_num) + '\t' + class_name + '\t' + str(ratio))

        if len(vocabulary) <= min_vocab:
            break

    print('Final Ratio\t' + class_name + '\t' + str(ratio))
    #
    for id_ in partition1:
        if id_ >= num_words:
            text_string = data_list[id_ - num_words]
            # t = tokenizer.tokenize(text_string, return_str=True)
            t = text_string
            t = t.strip().replace('\t', ' ')
            fp_train.write(str(i) + '\t' + t + '\n')

    for id_ in partition2:
        if id_ >= num_words:
            text_string = data_list[id_ - num_words]
            # t = tokenizer.tokenize(text_string, return_str=True)
            t = text_string
            t = t.strip().replace('\t', ' ')
            fp_test.write(str(i) + '\t' + t + '\n')

fp_train.close()
fp_test.close()
