"""
1. This code trains a Multi-class Naive Bayes classifier on Arxiv abstracts dataset
2. Only the primary labels are used
3. Top features for every class are written to a file
"""

import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from stopwords import ENGLISH_STOP_WORDS


# Train Data path
train_text_path = "/home/dsachan/src/sequence_classification/arxiv_abstracts/arxiv_abstracts_data.txt"

# Output Features File
output_file = "/home/dsachan/src/sequence_classification/dataset_construction/nb_features_arxiv.txt"
output_dir = "/home/dsachan/src/sequence_classification/dataset_construction/arxiv_lexicons/{}"

# Valid Classes file
arxiv_classes = "../arxiv_abstracts/prim_cat_selected.txt"
# arxiv_classes = "../arxiv_abstracts/all_cat_selected.txt"

topk = 600
max_features = 150

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


def contains_stopword(x):
    tokens = x.split()
    for tok in tokens:
        if tok in ENGLISH_STOP_WORDS:
            return False
    else:
        return True


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
y_train = np.array(map(int, train_label))

"""
num_features_123 = 700000
counter_123 = Counter()

for num, sent in enumerate(train_text):
    if num % 50000 == 0:
        print num

    if num % 300000 == 0:
        counter_123 = Counter(dict(counter_123.most_common(4 * num_features_123)))

    # for feature_extractor in (unigram_features, bigram_features, trigram_features):
    for feature_extractor in (bigram_features, trigram_features, fourgram_features):
        t1 = feature_extractor(sent)
        counter_123.update(t1)


vocabulary_123 = counter_123.most_common(num_features_123)
vocabulary = vocabulary_123

del counter_123

with open('/home/dsachan/arxiv/vocab_2-4grams.txt', 'wb') as fp:
    for word, freq in vocabulary:
        fp.write(word + '\t' + str(freq) + '\n')
"""

vocabulary = []
vocab_freq = dict()
with open('/home/dsachan/arxiv/vocab_2-4grams.txt') as fp:
    for line in fp:
        word, freq = line.strip().split('\t')
        vocabulary.append(word.strip())
        vocab_freq[word] = int(freq)


print("Doing TfIdf Vectorization")
vectorizer = TfidfVectorizer(lowercase=True, use_idf=True, ngram_range=(2, 4),
                             norm=u'l2', vocabulary=vocabulary)

x_train = vectorizer.fit_transform(train_text)
feature_names = np.asarray(vectorizer.get_feature_names())

print("Training Naive Bayes classifier from scikit-learn")
clf = MultinomialNB(fit_prior=False)
clf.fit(x_train, y_train)

predicted_train = clf.predict(x_train)

print("The accuracy of this classifier on train data is : {}".format(str(accuracy_score(y_train, predicted_train))))
print(classification_report(y_train, predicted_train))

# Computing the most informative features for Naive Bayes Classification
# http://www.nltk.org/_modules/nltk/classify/naivebayes.html

# Since the probabilities are in log-scale, we can do:
coef = clf.coef_ - np.min(clf.coef_, axis=0, keepdims=True)

for j in range(len(coef)):
    with open(output_dir.format(id2class[j]), 'w') as fp:
        topk_indice = np.argsort(coef[j])[-topk:]
        topk_coeff = np.sort(coef[j])[-topk:]
        # fp.write("\n Top features for category : {}\n".format(id2class[j]))

        count_max_features = 0
        for word, score in zip(feature_names[topk_indice][::-1], topk_coeff[::-1]):
            flag = contains_stopword(word)
            if flag is True and vocab_freq[word] < 5000:
                count_max_features += 1
                fp.write(word + '\t' + str(vocab_freq[word]) + '\t' + str(score) + '\n')

                if count_max_features >= max_features:
                    break
