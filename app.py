from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

file = open('tfidf.pkl', 'rb')
tfidf = pickle.load(file)

file1 = open('model.pkl', 'rb')
classifier = pickle.load(file1)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)





