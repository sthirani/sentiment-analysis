from CleanData import *
from TrainData import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
test_df = load_dataset("/Users/thirani/Projects/Sentiment Analysis/addidas_tweets.csv", ['target', 'text'])

# clean dataset
test_df.text = test_df['text'].apply(preprocess_tweet_text)
print(test_df)
cv = CountVectorizer()
df_new = test_df['text']
print(df_new.tolist())
x_test = cv.fit_transform(df_new.tolist()).toarray()
print(x_test)
np.savetxt("/Users/thirani/Projects/Sentiment Analysis/demo_x.csv", x_test, delimiter=',', fmt='%s')

# tf_vector = get_feature_vector(np.array(test_df['text']).ravel())
# x_test = tf_vector.transform(np.array(test_df['text']).ravel())
print("-----", x_test.shape)
trained_model = pickle.load(open(trained_model_filename, 'rb'))
y_test = trained_model.predict(x_test)
np.savetxt("demo_y.csv", y_test, delimiter=',', fmt='%s')
print(y_test)
