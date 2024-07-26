# Twitter Sentiment Analysis

This project is a sentiment analysis system for Twitter data. It preprocesses tweets, trains a machine learning model, and serves predictions via a Flask API.

## Setup

1. Install dependencies:
```
pip install pandas nltk scikit-learn flask
```


2. Preprocess the data:
```
python preprocess.py
```


3. Train the model:
```
python train_model.py
```


4. Run the Flask app:
```
python app.py
```


## Usage

Send a POST request to `/predict` with JSON data:
```json
{
"tweet": "I love this product!"
}
