# sentiment-analysis

_Analyze sentiment of a specific brand using twitter data_

### Install python and pycharm

- https://www.python.org/downloads/
- https://www.jetbrains.com/pycharm/download/


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

- pre-process data by removing `#`, `@`,`/,.$"\` etc

````
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet) 
````
- visualize the twitter data using `Wordcloud`
```
 wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='white',
                          stopwords = stopwords,
                          min_font_size = 10).generate(words)

```
- analyze data to determine sentiment using `VADER`

````
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
````

- Example results on wordcloud and the overall sentiment is attached
- Tips & Tricks
    - Can save data in file
    ````
  np.savetxt("ans.csv", y_test, delimiter=',', fmt='%s') 
    ````
    - run DisableSSL.py to download nltk modules if necessary