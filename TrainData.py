from CleanData import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle

trained_model_filename = '/Users/thirani/Projects/Sentiment Analysis/trained_model.sav'


def vectorize(dataset):
    # Same tf vector will be used for Testing sentiments on unseen trending data
    # cv = CountVectorizer()
    # x = cv.fit_transform(dataset['text']).toarray()
    tf_vector = get_feature_vector(np.array(dataset['text']).ravel())
    x = tf_vector.transform(np.array(dataset['text']).ravel())
    y = np.array(dataset['sentiment'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

    return x_train, x_test, y_train, y_test


def train_model(x_train, x_test, y_train, y_test):
    # Training Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    y_predict_nb = nb_model.predict(x_test)
    print(accuracy_score(y_test, y_predict_nb))

    return nb_model


# def main():
x = pd.read_csv("/Users/thirani/Projects/Sentiment Analysis/clean_sentences.csv", encoding='latin-1')
print(x)
    # dataset = clean_dataset()
    # x_train, x_test, y_train, y_test = vectorize(dataset)
    # print(x_train.shape)
    # trained_model = train_model(x_train, x_test, y_train, y_test)
    # pickle.dump(trained_model, open(trained_model_filename, 'wb'))

#     # Load dataset
#     test_df = load_dataset("/Users/thirani/Projects/Sentiment Analysis/addidas_tweets.csv", ['target', 'text'])
#     # print(test_df['text'])
# # clean dataset
#     test_df.text = test_df['text'].apply(preprocess_tweet_text)
#     # print(test_df['text'])
#
#     # tf_vector = get_feature_vector(np.np.array(dataset.iloc[:, 1]))
#     # x_test = tf_vector.transform(np.array())
#     cv = CountVectorizer()
#     x_test = cv.fit_transform(np.array(test_df.text)).toarray()
#     print("-----", x_test.shape)
#     np.savetxt("/Users/thirani/Projects/Sentiment Analysis/demo_x.csv", x_test, delimiter=',', fmt='%s')
#
#
#     # trained_model = pickle.load(open(trained_model_filename, 'rb'))
#     y_test = trained_model.predict(x_test)
#     # np.savetxt("/Users/thirani/Projects/Sentiment Analysis/demo_y.csv", y_test, delimiter=',', fmt='%s')
#     print(y_test)



