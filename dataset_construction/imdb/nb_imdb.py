import os
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy


# Train Data path
train_text_path = "/home/dsachan/imdb/train.text"
train_label_path = "/home/dsachan/imdb/train.label"

# Test Data path
test_text_path = "/home/dsachan/imdb/test.text"
test_label_path = "/home/dsachan/imdb/test.label"

# Output Features File
output_dir = "imdb_lexicons/{}"

id2class = ['most_negative', 'negative', 'neutral', 'positive', 'most_positive']
topk = 1500


def read_data(loc_):
    data = list()
    with open(loc_) as fp:
        for line in fp:
            t = line.strip()
            data.append(t)
    return data


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
train_text = read_data(train_text_path)
y_train = np.array(map(int, read_data(train_label_path)))

test_text = read_data(test_text_path)
y_test = np.array(map(int, read_data(test_label_path)))

y_train = map(lambda x: np.ceil(x/2.), y_train)
y_test = map(lambda x: np.ceil(x/2.), y_test)

y_train = np.concatenate([y_train, y_test])

"""
num_features_123 = 400000
num_features_45 = 100000

counter_123 = Counter()
counter_45 = Counter()

for num, sent in enumerate(train_text + test_text):
    if num % 50000 == 0:
        print num

    if num % 300000 == 0:
        counter_123 = Counter(dict(counter_123.most_common(4 * num_features_123)))
        counter_45 = Counter(dict(counter_45.most_common(4 * num_features_45)))

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

with open('/home/dsachan/imdb/vocab_1-5grams.txt', 'wb') as fp:
    for word, freq in vocabulary:
        fp.write(word + '\t' + str(freq) + '\n')
"""

vocabulary = []
vocab_freq = dict()
with open('/home/dsachan/imdb/vocab_1-5grams.txt') as fp:
    for line in fp:
        word, freq = line.strip().split('\t')
        vocabulary.append(word.strip())
        vocab_freq[word] = int(freq)


print("Doing TfIdf Vectorization")
vectorizer = TfidfVectorizer(lowercase=True, use_idf=True, ngram_range=(1, 5),
                             norm=u'l2', vocabulary=vocabulary)

x_train = vectorizer.fit_transform(train_text + test_text)
# x_test = vectorizer.transform(test_text)
feature_names = np.asarray(vectorizer.get_feature_names())

print("Training Multi-Class Naive Bayes classifier from scikit-learn")
clf = MultinomialNB(fit_prior=False)
clf.fit(x_train, y_train)

predicted_train = clf.predict(x_train)

print("The accuracy of this classifier on train data is : {}".format(str(accuracy_score(y_train, predicted_train))))
print(classification_report(y_train, predicted_train))

# predicted_test = clf.predict(x_test)
# print("The accuracy of this classifier on test data is : {}".format(str(accuracy_score(y_test, predicted_test))))
# print(classification_report(y_test, predicted_test))


# Computing the most informative features for Naive Bayes Classification
# http://www.nltk.org/_modules/nltk/classify/naivebayes.html
# Since the probabilities are in log-scale, we can do:
coef = clf.coef_ - np.min(clf.coef_, axis=0, keepdims=True)


for j in range(len(coef)):
    with open(output_dir.format(id2class[j]), 'w') as fp:
        topk_indice = np.argsort(coef[j])[-topk:]
        topk_coeff = np.sort(coef[j])[-topk:]
        for word, score in zip(feature_names[topk_indice][::-1], topk_coeff[::-1]):
            fp.write(word + '\t' + str(vocab_freq[word]) + '\t' + str(score) + '\n')
