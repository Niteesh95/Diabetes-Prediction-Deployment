# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:45:56 2020

@author: Niteesh
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

pickle_in = open('diabetes-prediction-rfc-model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/predict_file', methods=["POST"])
def predict_diabeties_file():
    """
    for rendering result on HTML GUI
    """
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = classifier.predict(final_features)
    if prediction:
        return render_template('index.html', prediction_txt='Oops! You have diabetes.') 
    else:
        return render_template('index.html', prediction_txt="Great! You don't have diabetes.") 

if __name__ == '__main__':
    app.run(debug=True)