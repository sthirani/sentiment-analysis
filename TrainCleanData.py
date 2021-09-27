from CleanData import *

dataset = import_dataset("/Users/thirani/Projects/Sentiment Analysis/training_tweet_dataset.csv",
                         ['target', 'id', 'date', 'flag', 'user', 'text'])
ds_train = remove_unwanted_cols(dataset, ['id', 'date', 'flag', 'user'])
ds_train['target'] = ds_train['target'].replace(4,1)

ds_train_pos = ds_train[ds_train['target'] == 1]
ds_train_neg = ds_train[ds_train['target'] == 0]
ds_train_neu = ds_train[ds_train['target'] == 2]

ds_train_pos = ds_train_pos.iloc[:int(3000)]
ds_train_neg = ds_train_neg.iloc[:int(3000)]
ds_train_neu = ds_train_neu.iloc[:int(3000)]

dataset_train = pd.concat([ds_train_pos, ds_train_neg, ds_train_neu])
dataset_train.text = dataset_train["text"].apply(preprocess_tweet_text)
print(dataset_train)


