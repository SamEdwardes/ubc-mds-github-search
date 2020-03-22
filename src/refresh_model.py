import pandas as pd
import numpy as np
import os
import re
import pickle

import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from helpers import print_break
from model import train_model, find_query_weights, cos_similarity, most_similar


def update_model(X_train_path):
    print_break("Updating model")
    df = pd.read_csv(X_train_path)
    X_train = df.loc[:,"content"]
    tfid_vectorizer, X_train_weights = train_model(df["content"])
    tfid_vectorizer.stop_words = None # to reduce file size
    pickle.dump(tfid_vectorizer, open("data/model.pkl", 'wb'))
    scipy.sparse.save_npz('data/model_sparse_matrix.npz', X_train_weights)

def test_model(X_train_path):
    print_break("Testing model")
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
    update_model(X_train_path="data/student-repos.csv")
    test_model(X_train_path="data/student-repos.csv")
    