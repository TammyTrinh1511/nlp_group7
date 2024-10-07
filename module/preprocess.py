import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import nltk

import numpy as np

nltk.download('twitter_samples')
nltk.download('stopwords')


class Preprocessor:
    def __init__(self):
        self.tokenizer = TweetTokenizer(
            preserve_case=False, strip_handles=True, reduce_len=True)
        self.stemmer = PorterStemmer()
        self.stopwords_english = stopwords.words('english')

    def process_tweet(self, tweet: str):
        tweet = re.sub(r'\$\w*', '', tweet)
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
        tweet = re.sub(r'#', '', tweet)

        tweet_tokens = self.tokenizer.tokenize(tweet)
        tweets_clean = [self.stemmer.stem(word) for word in tweet_tokens
                        if word not in self.stopwords_english and word not in string.punctuation]
        return tweets_clean

    def preprocess_data(self, all_positive_tweets, all_negative_tweets):
        train_pos = [self.process_tweet(tweet)
                     for tweet in all_positive_tweets[:4000]]
        test_pos = [self.process_tweet(tweet)
                    for tweet in all_positive_tweets[4000:]]
        train_neg = [self.process_tweet(tweet)
                     for tweet in all_negative_tweets[:4000]]
        test_neg = [self.process_tweet(tweet)
                    for tweet in all_negative_tweets[4000:]]

        train_x = train_pos + train_neg
        test_x = test_pos + test_neg
        y_train = np.append(np.ones((len(train_pos), 1)),
                            np.zeros((len(train_neg), 1)), axis=0)
        y_test = np.append(np.ones((len(test_pos), 1)),
                           np.zeros((len(test_neg), 1)), axis=0)
        return train_x, test_x, y_train, y_test
