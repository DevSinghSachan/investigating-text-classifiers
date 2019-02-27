from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx


def create_data_partitions(data_list, vocab):

    edge_list = []
    vectorizer = TfidfVectorizer(lowercase=True,
                                 use_idf=False,
                                 # ngram_range=(2, 3), # For arxiv dataset
                                 ngram_range=(1, 5),   # For Imdb dataset
                                 norm=u'l2',
                                 vocabulary=vocab)

    x_train = vectorizer.fit_transform(data_list)

    # y_ind correspond to words in vocabulary
    # x_ind correspond to documents indices
    # In this graph, Node indices of words are from [0 to n_words-1] and
    #                Document indices are from [n_words to n_words + n_docs - 1]

    words_list = vocab
    num_words = len(words_list)
    num_nodes = num_words + len(data_list)
    x_indices, y_indices = x_train.nonzero()

    # Creating Graph edges
    for x_ind, y_ind in zip(x_indices, y_indices):
        edge_list.append((x_ind + len(words_list), y_ind))

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)

    # Creating two partitions according to Connected Components:
    cc_list = list()
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        cc_list.append(c)

    partiton1 = cc_list[0]  # Taking the largest Connected Component
    partiton2 = set.union(*cc_list[1:])  # Merging all the CC from 2nd smallest CC onwards
    ratio = len(partiton2) / float(len(partiton1))

    return partiton1, partiton2, ratio
