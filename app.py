# coding=utf-8

import numpy as np
import pandas as pd
import joblib
from itertools import chain
from collections import Counter
from flask import Flask, render_template, request, jsonify


import spacy
import en_core_web_sm

from gensim import models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('wordnet')
nltk.download('vader_lexicon')


# Load saved models
vectorizer = joblib.load('models/vectorizer.pkl')
classifier = joblib.load('models/classifier.pkl')
nmf_fn = joblib.load('models/nmf_fn.pkl')
nmf_kl = joblib.load('models/nmf_kl.pkl')
lda_model = models.LdaModel.load('models/lda.model')

# create SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# load the nlp model for POS and dependency tagging
nlp = spacy.load("en_core_web_sm")

# Array of aspects names for class prdiction, using supervised classification
aspects = ['food', 'menu', 'service', 'place', 'price', 'miscellaneous',
           'staff', 'ambience']


# Mappings between topics index and aspects using topic modeling methods
ldaTopic_to_aspect = {
    0: [],
    1: ['service'],
    2: ['staff'],
    3: [],
    4: ['service', 'ambience'],
    5: ['menu'],
    6: ['price'],
    7: []
}

nmf_fnTopic_to_aspect = {
    0: [],
    1: ['staff'],
    2: ['menu'],
    3: ['place'],
    4: [],
    5: [],
    6: ['staff'],
    7: ['service']
}

nmf_klTopic_to_aspect = {
    0: ['price', 'food'],
    1: ['staff'],
    2: ['menu'],
    3: ['place', 'ambience'],
    4: [],
    5: [],
    6: [],
    7: ['service']
}


app = Flask(__name__)


def preprocess_text(text, vectorizer=None):
    '''Preprocess a text for topic modeling or aspect identification'''
    if vectorizer:  # used for NMF and supervised model
        processed = vectorizer.transform([text])
    else:  # used for LDA
        text = remove_stopwords(text)
        tokens = simple_preprocess(text)
        tokens_lem = [WordNetLemmatizer().lemmatize(token, pos='v')
                      for token in tokens]
        processed = [token for token in tokens_lem if len(token) > 3]
    return processed


def get_main_topic(processed_text, model, topics_names, model_type='NMF'):
    ''' Predict the dominant topic in a text'''
    if model_type == 'LDA':
        bow_vector = model.id2word.doc2bow(processed_text)
        topics_scores = model[bow_vector]
        dominant_topic, _ = sorted(topics_scores,
                                   key=lambda tup: tup[1],
                                   reverse=True)[0]
    else:
        scores = model.transform(processed_text)
        scores = scores.reshape(-1)
        dominant_topic = np.argsort(scores)[-1]
    main_topic = topics_names[dominant_topic]
    if len(main_topic) == 0:
        main_topic = ['miscellaneous']
    return main_topic


def get_sentiment(text):
    '''return the sentiment. ouput: positive/negative/neutral'''
    scores = sia.polarity_scores(text)
    if scores['compound'] > 0:
        return 'positive'
    elif scores['compound'] < 0:
        return 'negative'
    else:
        return 'neutral'


def get_descriptors(text):
    '''get adjectives for a subject/noun'''

    def is_adjective(token):
        '''return whether or not a token is an adjective.'''
        return token.dep_ == 'amod' or token.pos_ == 'ADJ'

    def get_children(token):
        '''get a list reprsenting all adjectives that are either first 
        or second order children of a given token.'''
        first_ch = [child for child in token.children
                    if child.pos_ not in ['AUX', 'VERB']]
        second_ch = [list(ch.children) for ch in first_ch]
        second_ch = list(chain.from_iterable(second_ch))  # convert to 1D list
        return first_ch + second_ch

    subjects_descriptors = {}
    for token in nlp(text):
        # adjectives for subjects
        if token.dep_ == 'nsubj' and token.pos_ != 'PRON':
            descriptors = []
            # descriptive adjectives
            adjectives = [child for child in get_children(token)
                          if is_adjective(child)]
            descriptors.extend(adjectives)
            # predicate adjectives (using a linking verb)
            if token.head.pos_ in ['AUX', 'VERB']:
                descriptors.extend([child for child in get_children(token.head)
                                    if is_adjective(child)])
            descriptors = list(set(descriptors))
            subjects_descriptors[token] = descriptors
        # adjectives for non-subject nouns
        elif token.pos_ in ['NOUN', 'PROPN']:
            subjects_descriptors[token] = [child for child in get_children(token)
                                           if is_adjective(child)]
        else:
            continue
    return subjects_descriptors


app = Flask(__name__)


# home webpage
@app.route('/')
def index():
    return render_template('index.html')


# results webpage: analysis for user input
@app.route('/analysis', methods=['POST'])
def analysis():
    text = request.form['review']
    phrases = [txt.strip() for txt in text.split('.') if len(txt) > 0]
    output = {}
    for phrase in phrases:
        # Text processing
        preprocessed = preprocess_text(phrase, vectorizer=vectorizer)
        preprocessed_lda = preprocess_text(phrase, vectorizer=None)

        # identify aspects using all models and concatenate them
        topics_full = []
        # supervised model
        label = classifier.predict(preprocessed)
        aspects_pred = [aspects[i] for i, is_predicted
                        in enumerate(label.reshape(-1)) if is_predicted]
        topics_full.extend(aspects_pred)
        # NMF with Frobenius norm
        topics = get_main_topic(preprocessed, nmf_fn, nmf_fnTopic_to_aspect,
                                model_type='NMF')
        topics_full.extend(topics)
        # NMF with Kullback-Leibler divergence
        topics = get_main_topic(preprocessed, nmf_kl, nmf_klTopic_to_aspect,
                                model_type='NMF')
        topics_full.extend(topics)
        # LDA
        topics = get_main_topic(preprocessed_lda, lda_model, ldaTopic_to_aspect,
                                model_type='LDA')
        topics_full.extend(topics)
        # Select topic with highest vote
        counts = Counter(topics_full)
        main_topics = [topic for (topic, count) in counts.items()
                       if counts[topic] == max(counts.values())]

        # Sentiment analysis
        sentiment = get_sentiment(phrase)

        # Find aspect descriptors
        subjects_descriptors = get_descriptors(phrase)
        subjects = sorted(set(subjects_descriptors.keys()))
        descriptors = sorted(
            set(chain.from_iterable(subjects_descriptors.values()))
        )

        output[phrase] = [main_topics, sentiment, subjects, descriptors]

    # pd.set_option('colheader_justify', 'left')
    df = pd.DataFrame.from_dict(output, orient='index',
                                columns=['Topic', 'Sentiment',
                                         'Discussion Subject', 'Descriptors'])

    return render_template('analysis.html',
                           table=[df.to_html(justify='left', bold_rows=False,
                                             classes=["table-striped"])])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
