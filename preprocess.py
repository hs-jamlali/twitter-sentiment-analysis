import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = word_tokenize(tweet)
    tweet = [word for word in tweet if word not in stopwords.words('english')]
    return ' '.join(tweet)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['tweet'] = df['tweet'].apply(preprocess_tweet)
    return df

if __name__ == "__main__":
    data = preprocess_data('data/sample_tweets.csv')
    data.to_csv('data/preprocessed_tweets.csv', index=False)
    print("Data preprocessed and saved to data/preprocessed_tweets.csv")
