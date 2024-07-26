from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data['tweet']
    tweet_vector = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vector)
    return jsonify({'label': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
