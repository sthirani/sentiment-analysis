#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Global Parameters

meaningful_words = set(nltk.corpus.words.words())
remove_tagged_words = ['NN', 'VB', 'PRP', 'RP', 'WDT', 'IN', 'DT', 'PDT', 'PRP$', 'WP', 'CD', 'CC', 'SYM']
additional = ['rt', 'rts', 'retweet', "b'RT", 'b']
stop_words = set().union(stopwords.words('english'), additional)
stopwords = set(STOPWORDS)


def import_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def plot_wordcloud(words):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def preprocess_tweet_text(tweet):
    tweet.lower()
    # remove retweets
    tweet = re.sub(r"b'rt+|b'+|b'RT", "", tweet)
    # remove hexa
    tweet = re.sub(r'(\\x(.){2})|x(.){2}', '', tweet)
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references from tweet
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)
    # Replace # with _ sign
    tweet = tweet.replace("#", "").replace("_", " ")
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # type caste each val to string
    val = str(tweet)
    # split the value
    tokens = val.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    # tokenize tweets
    tweet_tokens = word_tokenize(tweet)
    tagged_tweets = pos_tag(tokens)
    # remove nouns pronouns determiners, prepositions etc
    useful_tweets = [w for (w, y) in tagged_tweets if y not in remove_tagged_words]
    # Remove empty List from List
    non_empty_tweets = [w for w in useful_tweets if w != []]
    # Remove stop_words
    filtered_words = [w for w in non_empty_tweets if w not in stop_words]
    return " ".join(filtered_words) + " "


def add_label(tweet):
    tweet_tokens = word_tokenize(tweet)
    for tweet in tweet_tokens:
        analysis = TextBlob(tweet)
        return analysis.sentiment.polarity


def polarity_label(sentiment):
    if sentiment < 0.0:
        return 0
    elif sentiment == 0.0:
        return 1
    else:
        return 2


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


def clean_dataset(comment_words=''):
    dataset = import_dataset("/Users/thirani/Projects/Sentiment Analysis/addidas_tweets.csv",
                             ['target', 'text'])

    dataset.text = dataset["text"].apply(preprocess_tweet_text)
    for w in dataset.text:
        comment_words += w
    plot_wordcloud(comment_words)

    save_array = np.array(dataset['text'])

    np.savetxt("/Users/thirani/Projects/Sentiment Analysis/clean_sentences_nike.csv", save_array, delimiter=',',
               fmt='%s')


if __name__ == "__main__":
    clean_dataset()
