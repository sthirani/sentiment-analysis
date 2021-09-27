import numpy as np
from CleanData import import_dataset
from nltk.sentiment.vader import SentimentIntensityAnalyzer

dataset = import_dataset("/Users/thirani/Projects/Sentiment Analysis/clean_sentences.csv",
                         ['sentence'])

sentences = np.array(dataset.sentence)
analyzer = SentimentIntensityAnalyzer()
result = {'pos': 0, 'neg': 0, 'neu': 0}
for sentence in sentences:
    scores = analyzer.polarity_scores(sentence)
    if scores['compound'] > 0.05:
        result['pos'] += 1
    elif scores['compound'] < -0.05:
        result['neg'] += 1
    else:
        result['neu'] += 1
print(result)

if result['pos'] > result['neg']:
    print("Social media is feeling positive about nike")
else:
    print("Social media is feeling negative about nike")