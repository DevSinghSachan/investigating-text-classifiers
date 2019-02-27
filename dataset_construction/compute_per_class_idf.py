import numpy as np
from collections import Counter
import re

topk = 500
num_classes = 5
out_file = "/home/dsachan/imdb/idf/{}.txt"

vocabulary = []
fp = open('/home/dsachan/imdb/vocab_1-5grams.txt')
for word in fp:
    vocabulary.append(word.strip())
fp.close()

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

def ngram_features(text, n=2):
    """Produces n-gram features joined by __."""
    ngrams = []
    words = token_pattern.findall(text.lower())
    for i in range(len(words) - (n - 1)):
        ngrams.append(NGRAM_JOINER.join(words[i:(i + n)]))
    return ngrams


def read_data(loc_):
    data = list()
    with open(loc_) as fp:
        for line in fp:
            t = line.strip()
            data.append(t)
    return data


def idf_times_logtf(data_list):
    num_features = 500000
    N = float(len(data_list))
    counter_df = Counter()
    counter_tf = Counter()

    for num, sent in enumerate(data_list):
        if num % 50000 == 0:
            print num

        #if num % 300000 == 0:
        #    counter_df = Counter(dict(counter_df.most_common(4 * num_features)))
        #    counter_tf = Counter(dict(counter_tf.most_common(4 * num_features)))

        for feature_extractor in (unigram_features, bigram_features, trigram_features, fourgram_features):
            t = feature_extractor(sent)
            counter_tf.update(t)
            counter_df.update(set(t))

    vocab = counter_tf.most_common(num_features)

    feature_dict = dict()
    for w, f in vocab:
        # feature dict = IDF * log(TF)
        feature_dict[w] = np.log10(N / counter_df[w]) * np.log10(f)

    sorted_list = sorted(feature_dict.iteritems(), key=lambda x: x[1], reverse=True)
    return sorted_list



# Train Data path
train_text_path = "/home/dsachan/imdb/train.text"
train_label_path = "/home/dsachan/imdb/train.label"

print("Reading imdb reviews")
train_text = read_data(train_text_path)
y_train = np.array(map(int, read_data(train_label_path)))
y_train = map(lambda x: np.ceil(x/2), y_train)

#    train_text = train_text[:100000]
#    y_train = y_train[:100000]


for i in range(1, num_classes + 1):
    data_list = []
    for x, y in zip(train_text, y_train):
        if y == i:
            data_list.append(x)

    feat_list = idf_times_logtf(data_list)
    with open(out_file.format(i), 'w') as fp:
        for w, s in feat_list:
            fp.write(w + '\t' + str(s) + '\n')
