import pandas as pd
import requests
import numpy as np
import pickle
import re
import json
import string
import matplotlib.pyplot as plt
import wordcloud
import seaborn as sns

from gensim.models import Word2Vec

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, BaseEstimator, BaseNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.collocations import *
import nltk 

import pkg_resources
from symspellpy.symspellpy import SymSpell

import pickle



# gets text from a gutenberg URL
def get_guten(url):
    # retrieve the source text
    r = requests.get(url)
    r.encoding = 'utf-8'
    text = r.text
    return text

# gets the text from a txt file
def get_text(path, encoding='utf8'):
    f = open(path, 'r', encoding=encoding)
    text = f.read()
    f.close()
    return text

# turns the text into a series of sentences and standardizes things like case
# also cleans text by removing numbers and punctuation
def tokenize_text(text):
    quotes = sent_tokenize(text)
    cleaned_sentences = []
    for quote in quotes:
        quote = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff\xad\x0c6§\[\]\\\£\Â\n\r]', '', quote)
        quote = re.sub(r'[0123456789]', ' ', quote)
        for char in string.punctuation:
            quote = quote.replace(char, ' ').replace(' asa ', ' as a ')
        cleaned_sentences.append(quote)
    # remove unnecessary spaces
    cleaned_sentences = [x.strip() for x in cleaned_sentences]
    # remove empty quotes
    cleaned_sentences = list(filter(None, cleaned_sentences))
    # cut out very short ones as they often have no real meaning
    cleaned_sentences = [x for x in cleaned_sentences if len(x) > 10]
    # remove the titles of sections & citation-type stuff
    cleaned_sentences = [x for x in cleaned_sentences if not x.isupper()]
    cleaned_sentences = [x for x in cleaned_sentences if not x.replace('the', '').replace('of', '').replace('and', '').replace('II', '').istitle()]
    cleaned_sentences = [x.lower() for x in cleaned_sentences]
    return cleaned_sentences

def make_word_cloud(text, stopwords=stopwords.words('english')):
    cloud = wordcloud.WordCloud(width=2000, 
                            height=1100, 
                            background_color='#D1D1D1', 
                            max_words=30, 
                            stopwords=stopwords, 
                            color_func=lambda *args, **kwargs: (95,95,95)).generate(text)
    return cloud

def plot_pretty_cf(predictor, xtest, ytest, cmap='Greys', normalize='true', title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confusion_matrix(predictor, xtest, ytest, cmap=cmap, normalize=normalize, ax=ax)
    ax.set_title(title, size='xx-large', pad=20, fontweight='bold')
    ax.set_xticklabels([str(x).replace('_', ' ').title()[12:-2] for x in ax.get_xticklabels()], rotation=35)
    ax.set_yticklabels([str(x).replace('_', ' ').title()[12:-2] for x in ax.get_yticklabels()])
    ax.set_xlabel('Predicted Label', size='x-large')
    ax.set_ylabel('True Label', size='x-large')
    plt.show()

def classify_text(to_classify, model, vectorizer, verbose=5):
    predictor_pipeline = make_pipeline(vectorizer, model) 
    class_names = ['analytic', 'continental', 'phenomenology', 'german_idealism', 'plato', 'aristotle', 'empiricism', 'rationalism']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(to_classify, predictor_pipeline.predict_proba, num_features=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])
    # label_dict = {}
    # for label in exp.as_map().keys():
    #     for pair in exp.as_map()[label]:
    #         classifier_sum = 0
    #         classifier_sum += abs(pair[1])
    #         label_dict[label] = classifier_sum
    # labels = sorted(label_dict, key=label_dict.get, reverse=True)[:verbose]
    exp.show_in_notebook(text=True)

def make_w2v(series, stopwords=None, size=200, window=5, min_count=5, workers=-1, epochs=20):
    sentences = series.map(word_tokenize)
    cleaned_sentences = []
    for sentence in list(sentences):
        cleaned = [x.lower() for x in sentence if x.lower() not in stopwords]
        cleaned_sentences.append(cleaned)
    model = Word2Vec(list(cleaned_sentences), size=size, window=window, min_count=min_count, workers=workers)
    model.train(cleaned_sentences, total_examples=model.corpus_count, epochs=epochs)
    model_wv = model.wv
    return model_wv
    
