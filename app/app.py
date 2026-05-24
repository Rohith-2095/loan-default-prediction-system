from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load(
    '../models/best_model.pkl'
)

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = []

    for value in request.form.values():

        try:

            features.append(float(value))

        except:

            features.append(0)

    # Ensure 15 features
    while len(features) < 15:

        features.append(0)

    features = np.array(
        features
    ).reshape(1, -1)

    prediction = model.predict(features)

    result = "Loan Approved"

    if prediction[0] == 0:

        result = "Loan Rejected"

    return render_template(
        'index.html',
        prediction_text=result
    )

if __name__ == '__main__':

    app.run(debug=True)