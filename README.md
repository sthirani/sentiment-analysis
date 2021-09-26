# sentiment-analysis

_Analyze sentiment of a specific brand using twitter data_

### Install python and jupyterlab

- https://www.python.org/downloads/

- sudo -H python -m ensurepip

- pip install jupyterlab

### Git commands to clone the project

- git clone git@github.com:sthirani/sentiment-analysis.git

## Project steps

- import dependencies

```
import tweepy
import csv
import pandas as pd
import re
import numpy as np
import nltk
...
```

- grabbing tweets using Twitter API

  - Authentication

  ```
  auth = tw.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tw.API(auth, wait_on_rate_limit=True)
  ```

  - collect tweet on a topic

  ```
  tweets = tw.Cursor(api.search,q=search_words,...)
  ```

  - write to a csv file

  ```
  csvWriter = csv.writer(csvFile)
  ```

- load the dataset

````
dataset = load_dataset("trump_tweets.csv", ['target','text']) ```
````

- pre-process data

````
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet) ```
````

- train data

````
X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.2, random_state=30)
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train) ```
````

- use trained model to analyze new dataset
