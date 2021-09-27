import tweepy
import csv

# twitter authenticate KEYS
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

# Variables
FILENAME = "*_tweets.csv"
PATH_TO_FILE = "/Users/thirani/Projects/Sentiment Analysis/"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN,
                      ACCESS_TOKEN_SECRET)

# twitter API
api = tweepy.API(auth, wait_on_rate_limit=True)

# Open/create a file to append data to
with open(PATH_TO_FILE + FILENAME, 'a') as csvFile:
    # Use csv writer
    csvWriter = csv.writer(csvFile)
    # search for tweet on a topic
    for tweet in tweepy.Cursor(api.search_30_day,
                               label="search",
                               query="*",
                               fromDate="202109200000",
                               maxResults=500,
                               ).items():
        if tweet.lang == "en":
            csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
            # Write a row to the CSV file.
            print(tweet.created_at, tweet.text)
