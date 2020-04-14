import pandas as pd
import numpy as np
import os
import re
import pickle

import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk.corpus import stopwords


def text_preprocess(x):
    """Preprocess text data
    
    Parameters
    ----------
    x : str
        Text data to preprocess
    
    Returns
    -------
    str
        Preprocessed text data
    """
    x = str(x)
    regex = re.compile('[^a-zA-Z ]')
    x = regex.sub('', x)
    x = ' '.join(x.split()) # remove all double or more white space
    x = ' '.join([w for w in x.split() if len(w) < 25]) # get rid of long words
    stop_words = set(stopwords.words('english'))
    blob = TextBlob(x)
    words = blob.words
    words = words.lower()
    words = words.singularize()
    words = [w for w in words if not w in stop_words]
    return " ".join(words)
    

def train_model(X_train):
    """[summary]
    
    Parameters
    ----------
    X_train : pandas.Series
        A single column where each row contains text for a file
    
    Returns
    -------
    (sklearn.feature_extraction.text.TfidfVectorizer, 
     sparse matrix, [n_samples, n_features])
        Tuple containing TFID matrix and the X_train weights
    """
    print(f"Preprocessing {len(X_train)} text files...")    
    X_train = X_train.apply(text_preprocess)
    tfid_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    print("Creating TFID matrix...")
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
    