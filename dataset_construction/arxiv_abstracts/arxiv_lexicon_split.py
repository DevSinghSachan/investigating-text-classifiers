# This script creates training and test dataset based on generated lexicons

import os
import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from graph_util import spectral_ordering
from util import create_data_partitions

min_vocab = 20

lexicons_dir = 'arxiv_lexicons'     # Lexicons directory
arxiv_classes = '../arxiv_abstracts/prim_cat_selected.txt'      # Valid Classes file
vocab_file = '/home/dsachan/arxiv/vocab_1-3grams.txt'
train_text_path = "/home/dsachan/src/sequence_classification/arxiv_abstracts/arxiv_abstracts_data.txt"   # Train Data path
arxiv_train = "/home/dsachan/src/sequence_classification/dataset_construction/arxiv_data_train_lexicon.txt"
arxiv_test = "/home/dsachan/src/sequence_classification/dataset_construction/arxiv_data_test_lexicon.txt"

num_classes = len(os.listdir(lexicons_dir))
fp_train = open(arxiv_train, 'w')
fp_test = open(arxiv_test, 'w')


# Read the arxiv classes
class2id = dict()
id2class = list()
with open(arxiv_classes) as fp:
    for line in fp:
        class_name = line.strip()
        class2id[class_name] = len(class2id)
        id2class.append(class_name)


def read_data(loc_):
    data = list()
    label = list()
    with open(loc_) as fp:
        for line in fp:
            try:
                pc, sc, title, summary = line.strip().split('\t')
                if pc in class2id:
                    text = title + ' ' + summary
                    data.append(text)
                    label.append(class2id[pc])
            except:
                continue
    return data, label


print("Reading text data")
train_text, train_label = read_data(train_text_path)
y_train = np.array(map(int, train_label))


# Read the lexicon files
lexicons_dict = defaultdict(list)
for category in os.listdir(lexicons_dir):
    path_ = lexicons_dir + '/' + category
    with open(path_) as fp:
        for line in fp:
            word = line.split('\t')[0]
            # lexicons_dict[category].append(vocab2id[word])
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
    while ratio < 0.6:
        loop_num += 1
        vocabulary = vocabulary[:-10]
        num_words = len(vocabulary)
        partition1, partition2, ratio = create_data_partitions(data_list, vocabulary)
        print(str(loop_num) + '\t' + class_name + '\t' + str(ratio))

        if len(vocabulary) <= min_vocab:
            break

    print('Final Ratio\t' + class_name + '\t' + str(ratio))
    #
    for id_ in partition1:
        if id_ >= num_words:
            fp_train.write(str(i) + '\t' + data_list[id_ - num_words] + '\n')

    for id_ in partition2:
        if id_ >= num_words:
            fp_test.write(str(i) + '\t' + data_list[id_ - num_words] + '\n')

        #for id_, val_ in zip(range(num_nodes), cluster_labels):
        # for id_, val_ in order:
        #    fp.write(str(id_) + '\t' + str(val_) + '\n')
        #for word in common_words_set:
        #    fp.write(word + '\n')

fp_train.close()
fp_test.close()