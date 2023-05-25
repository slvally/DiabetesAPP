from flask import Flask, request, render_template
import pickle
import pandas as pd 
import numpy as np 

app = Flask(__name__)

model_file = open('Diabetes.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    Pregnancies=float(request.form['Pregnancies'])
    
    Glucose=float(request.form['Glucose'])

    BloodPressure=float(request.form['BloodPressure'])

    SkinThickness=float(request.form['SkinThickness'])

    Insulin=float(request.form['Insulin'])

    BMI=float(request.form['BMI'])

    DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])

    Age=float(request.form['Age'])

    x=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

 
    
    prediction = model.predict(x)
    output = round(prediction[0],0)
    if (output==0):
        kelas="No Diabetes"
    else:
        kelas="Diabetes"


    return render_template('index.html', hasil=kelas, Pregnancies=Pregnancies, Glucose=Glucose, BloodPressure=BloodPressure, SkinThickness=SkinThickness, Insulin=Insulin, BMI=BMI, DiabetesPedigreeFunction=DiabetesPedigreeFunction, Age=Age)


if __name__ == '__main__':
    app.run(debug=True)