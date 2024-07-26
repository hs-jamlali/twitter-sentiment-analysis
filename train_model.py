import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle


def train_model(data_path):
    data = pd.read_csv(data_path)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['tweet'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer trained and saved to model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    train_model('data/preprocessed_tweets.csv')
