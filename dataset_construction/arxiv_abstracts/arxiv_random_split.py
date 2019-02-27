import os
import numpy as np
from sklearn.model_selection import train_test_split


lexicons_dir = 'arxiv_lexicons'     # Lexicons directory
arxiv_classes = '../arxiv_abstracts/prim_cat_selected.txt'      # Valid Classes file
train_text_path = "/home/dsachan/src/sequence_classification/arxiv_abstracts/arxiv_abstracts_data.txt"
arxiv_train = "/home/dsachan/src/sequence_classification/dataset_construction/arxiv_data_train.txt"
arxiv_test = "/home/dsachan/src/sequence_classification/dataset_construction/arxiv_data_test.txt"

num_classes = len(os.listdir(lexicons_dir))

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
data_list, label_list = read_data(train_text_path)
y_data = np.array(map(int, label_list))

print("Creating train-test split")
train_text, test_text, y_train, y_test = train_test_split(data_list, y_data, train_size=0.6)

print("Saving train data to file")
with open(arxiv_train, 'w') as fp:
    for l, x in zip(y_train, train_text):
        fp.write(str(l) + '\t' + x + '\n')

print("Saving test data to file")
with open(arxiv_test, 'w') as fp:
    for l, x in zip(y_test, test_text):
        fp.write(str(l) + '\t' + x + '\n')