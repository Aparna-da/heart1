import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math



app = Flask(__name__, template_folder='template')
model = pickle.load(open('heart.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html',prediction_text="heart disease 0 or 1  {}".format(math.floor(output)))



if __name__ == '__main__':
    app.run(debug=True)
