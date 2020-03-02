import pandas as pd
import numpy as np
import os
import re
import pickle

import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_preprocess(x):
    regex = re.compile('[^a-zA-Z ]')
    x = regex.sub('', x)
    x = x.lower()
    return x

def train_model(X_train):
    tfid_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,4))
    X_train_weights = tfid_vectorizer.fit_transform(X_train)
    return (tfid_vectorizer, X_train_weights)

def find_query_weights(X_query, tfid_vectorizer):
    X_query = text_preprocess(X_query)
    X_query_weights = tfid_vectorizer.transform([X_query])
    return X_query_weights

def cos_similarity(X_query_weights, X_train_weights):
    cosine_distance = cosine_similarity(X_query_weights, X_train_weights)
    similarity_list = cosine_distance[0]
    return similarity_list

def most_similar(similarity_list, min_hits=4):
    most_similar= []
    while min_hits > 0:
        tmp_index = np.argmax(similarity_list)
        most_similar.append(tmp_index)
        similarity_list[tmp_index] = 0
        min_hits -= 1
    return most_similar

def update_model(X_train_path):
    df = pd.read_csv(X_train_path)
    X_train = df.loc[:,"content"]

    tfid_vectorizer, X_train_weights = train_model(df["content"])
    pickle.dump(tfid_vectorizer, open("data/model.pkl", 'wb'))
    scipy.sparse.save_npz('data/model_sparse_matrix.npz', X_train_weights)

def test_model(X_train_path):
    df = pd.read_csv(X_train_path)
    tfid_vectorizer = pickle.load(open("data/model.pkl", "rb"))
    X_train_weights = scipy.sparse.load_npz('data/model_sparse_matrix.npz')

    search_query = "mle pandas"
    X_query_weights = find_query_weights(search_query, tfid_vectorizer)
    sim_list = cos_similarity(X_query_weights, X_train_weights)
    df["score"] = sim_list
    print(df.sort_values(by="score", ascending=False).head())
    X_query_index = most_similar(sim_list, 5)
    print(X_query_index)

if __name__ == '__main__':
    # update_model(X_train_path="data/2020-03-01_student-repos.csv")
    test_model(X_train_path="data/2020-03-01_student-repos.csv")