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
    