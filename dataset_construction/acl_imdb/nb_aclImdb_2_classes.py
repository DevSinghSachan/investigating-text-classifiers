import os
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy


# Train Data path
train_data_path = "/home/dsachan/imdb/aclImdb/train.txt"

# Test Data path
test_data_path = "/home/dsachan/imdb/aclImdb/test.txt"

# Output Features File
output_dir = "aclImdb_lexicons/{}"

id2class = ['negative', 'positive']
topk = 1500


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


print("Reading imdb reviews")
train_label, train_text = read_data(train_data_path)
y_train = np.array(map(int, train_label))

test_label, test_text = read_data(test_data_path)
y_test = np.array(map(int, test_label))

y_train = np.concatenate([y_train, y_test])

"""
num_features_123 = 200000
num_features_45 = 50000

counter_123 = Counter()
counter_45 = Counter()

for num, sent in enumerate(train_text + test_text):
    if num % 1000 == 0:
        print num

    for feature_extractor in (unigram_features, bigram_features, trigram_features):
        t1 = feature_extractor(sent)
        counter_123.update(t1)

    for feature_extractor in (fourgram_features, fivegram_features):
        t1 = feature_extractor(sent)
        counter_45.update(t1)

vocabulary_123 = counter_123.most_common(num_features_123)
vocabulary_45 = counter_45.most_common(num_features_45)
vocabulary = vocabulary_123 + vocabulary_45

del counter_123
del counter_45

with open('/home/dsachan/imdb/aclImdb/vocab_1-5grams.txt', 'wb') as fp:
    for word, freq in vocabulary:
        fp.write(word + '\t' + str(freq) + '\n')
"""

vocabulary = []
vocab_freq = dict()
with open('/home/dsachan/imdb/aclImdb/vocab_1-5grams.txt') as fp:
    for line in fp:
        word, freq = line.strip().split('\t')
        vocabulary.append(word.strip())
        vocab_freq[word] = int(freq)


print("Doing TfIdf Vectorization")
vectorizer = TfidfVectorizer(lowercase=True, use_idf=True, ngram_range=(1, 5),
                             norm=u'l2', vocabulary=vocabulary)

x_train = vectorizer.fit_transform(train_text + test_text)
feature_names = np.asarray(vectorizer.get_feature_names())

print("Training Multi-Class Naive Bayes classifier from scikit-learn")
clf = MultinomialNB(fit_prior=False)
clf.fit(x_train, y_train)
predicted_train = clf.predict(x_train)

print("The accuracy of this classifier on train data is : {}".format(str(accuracy_score(y_train, predicted_train))))
print(classification_report(y_train, predicted_train))


# Computing the most informative features for Naive Bayes Classification
# http://www.nltk.org/_modules/nltk/classify/naivebayes.html
# Since the probabilities are in log-scale, we can do:

coef = clf.feature_log_prob_ - np.min(clf.feature_log_prob_, axis=0, keepdims=True)

for j in range(len(coef)):
    with open(output_dir.format(id2class[j]), 'w') as fp:
        topk_indice = np.argsort(coef[j])[-topk:]
        topk_coeff = np.sort(coef[j])[-topk:]
        for word, score in zip(feature_names[topk_indice][::-1], topk_coeff[::-1]):
            fp.write(word + '\t' + str(vocab_freq[word]) + '\t' + str(score) + '\n')
