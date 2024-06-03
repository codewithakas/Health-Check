from flask import jsonify, render_template, request
from service import app
from service.brain import *

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    req = request.form.to_dict()
    data = []
    data.append(int(req['pregnancies']))
    data.append(float(req['glucose']))
    data.append(float(req['blood_pressure']))
    data.append(int(req['insulin']))
    data.append(float(req['bmi']))
    data.append(float(req['dpf']))
    data.append(int(req['age']))
    data = [data]
    predictions, accuracy = predict(data)
    if predictions == 1:
        prediction = 'Diabetic'
    elif predictions == 0:
        prediction = 'Not diabetic'
    final_data = {'prediction': str(prediction), 'accuracy': str(round(accuracy, 3)*100) + '%', 'prediction_numerical': str(predictions)}
    return render_template('results.html', data=final_data)
