import os
import re
import numpy as np
from collections import Counter
from gensim.models.tfidfmodel import TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Train Data path
train_text_path = "/home/dsachan/src/sequence_classification/arxiv_abstracts/arxiv_abstracts_data.txt"

# Output Features File
output_file = "/home/dsachan/src/sequence_classification/dataset_construction/nb_features_arxiv_all_classes.txt"

# Valid Classes file
# arxiv_classes = "../arxiv_abstracts/prim_cat_selected.txt"
arxiv_classes = "../arxiv_abstracts/all_cat_selected.txt"


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
                mul_label = []
                pc, sc, title, summary = line.strip().split('\t')
                flag1, flag2 = False, False
                sc_list = sc.split(',')
                if len(sc_list) > 0:
                    for sclass in sc_list:
                        if sclass in class2id:
                            flag1 = True
                            mul_label.append(class2id[sclass])

                if pc in class2id:
                    flag2 = True
                    mul_label.append(class2id[pc])

                if flag1 or flag2:
                    text = title + ' ' + summary
                    data.append(text)
                    # label.append(class2id[pc])
                    label.append(list(set(mul_label)))
            except:
                continue
    return data, label


# FEATURE FUNCTIONS
token_pattern = re.compile(r'(?u)\b\w\w+\b')


def unigram_features(text):
    return token_pattern.findall(text.lower())


# An arbitrary symbol, in case whitespace is confusing/problematic:
NGRAM_JOINER = " "


def bigram_features(text):
    return ngram_features(text, n=2)


def trigram_features(text):
    return ngram_features(text, n=3)


def fourgram_features(text):
    return ngram_features(text, n=4)


def fivegram_features(text):
    return ngram_features(text, n=5)


def ngram_features(text, n=2):
    """Produces n-gram features joined by __."""
    ngrams = []
    words = token_pattern.findall(text.lower())
    for i in range(len(words) - (n - 1)):
        ngrams.append(NGRAM_JOINER.join(words[i:(i + n)]))
    return ngrams


print("Reading arxiv data")
train_text, train_label = read_data(train_text_path)
y_train = MultiLabelBinarizer().fit_transform(train_label)


# y_train = np.array(map(int, train_label))

"""
num_features_123 = 700000
counter_123 = Counter()

for num, sent in enumerate(train_text):
    if num % 50000 == 0:
        print num

    if num % 300000 == 0:
        counter_123 = Counter(dict(counter_123.most_common(4 * num_features_123)))

    for feature_extractor in (unigram_features, bigram_features, trigram_features):
        t1 = feature_extractor(sent)
        counter_123.update(t1)


vocabulary_123 = counter_123.most_common(num_features_123)
vocabulary = vocabulary_123

del counter_123

fp = open('/home/dsachan/arxiv/vocab_1-3grams.txt', 'wb')
for word, freq in vocabulary:
    print >> fp, word
fp.close()
"""

vocabulary = []
fp = open('/home/dsachan/arxiv/vocab_1-3grams.txt')
for word in fp:
    vocabulary.append(word.strip())
fp.close()


print("Doing TfIdf Vectorization")
vectorizer = TfidfVectorizer(lowercase=True, use_idf=True, ngram_range=(1, 3),
                             norm=u'l2', vocabulary=vocabulary)

x_train = vectorizer.fit_transform(train_text)
feature_names = np.asarray(vectorizer.get_feature_names())

print("Training Naive Bayes classifier from scikit-learn")
clf = OneVsRestClassifier(MultinomialNB(fit_prior=False), n_jobs=1)
clf.fit(x_train, y_train)

predicted_train = clf.predict(x_train)

print("The accuracy of this classifier on train data is : {}".format(str(accuracy_score(y_train, predicted_train))))
print(classification_report(y_train, predicted_train))

topk = 200

# Computing the most informative features for Naive Bayes Classification
# http://www.nltk.org/_modules/nltk/classify/naivebayes.html

# coef = clf.coef_ / np.min(clf.coef_, axis=0, keepdims=True)

# Since the probabilities are in log-scale, we can do:
# coef = clf.coef_ - np.min(clf.coef_, axis=0, keepdims=True)

# import pdb; pdb.set_trace()
coef = clf.estimators_

"""
with open(output_file, 'w') as fp:
    for j in range(len(coef)):
        topk_indice = np.argsort(coef[j])[-topk:]
        topk_coeff = np.sort(coef[j])[-topk:]
        fp.write("\n Top features for category : {}\n".format(id2class[j]))
        for word, score in zip(feature_names[topk_indice][::-1], topk_coeff[::-1]):
            fp.write(word + '\t' + str(score) + '\n')
"""

with open(output_file, 'w') as fp:
    for j in range(len(coef)):
        topk_indice = np.argsort(coef[j].coef_[0])[-topk:]
        topk_coeff = np.sort(coef[j].coef_[0])[-topk:]
        fp.write("\n Top features for category : {}\n".format(id2class[j]))
        for word, score in zip(feature_names[topk_indice][::-1], topk_coeff[::-1]):
            fp.write(word + '\t' + str(score) + '\n')
