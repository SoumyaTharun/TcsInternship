# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:04:30 2022

@author: bsoum
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('Salary_Predict.pkl','rb'))
@app.route('/')
def home():
    return render_template('predictsalaryhome.html')
@app.route('/predict',methods=['POST'])
def predict():
   int_features = [float(x) for x in request.form.values()]
   
   final_features = [np.array(int_features)]
   prediction = model.predict(final_features)
   output =prediction.item()
   return render_template('salaryResult.html', prediction_text='The salary of an employee is {}'.format(output))
  # return render_template ('result.html',prediction_text="Congrats!!...You are eligible for a Salary of Rs.{}".format(output))
if __name__=='__main__':
    app.run(port=8000)