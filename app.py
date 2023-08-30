from flask import Flask,render_template,request
import numpy as np 
from pickle import load
import pandas as pd


sv_classifier_health = load(open('Models/sv_classifier_health.pkl','rb'))
scaler = load(open('Models/standard_scaler_health.pkl','rb'))

nb_classifier = load(open('Models/nb_classifier_disease.pkl','rb'))
columns = load(open('dataset/columns.pkl','rb'))
precaution_data = load(open('dataset/app_precaution_data.pkl','rb'))
description_data = load(open('dataset/app_description_data.pkl','rb'))
doctor_data = load(open('dataset/app_doctor_dataset.pkl','rb'))

app = Flask(__name__)


#####################################

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/form',methods=['GET','POST'])
def forms():
    if request.method == 'GET':
        return render_template('first_page.html')
    else: 
        c_name = request.form.get('name')
        c_gender = request.form.get('gender')
        c_age = request.form.get('age')
        c_heart_rate = request.form.get('heart_rate')
        c_temp = request.form.get('body_temp')
        c_spo2 = request.form.get('spo2')
        c_bpm = request.form.get('bpm')

        age = float(c_age)
        heart_rate = float(c_heart_rate)
        temp = float(c_temp)
        spo2 = float(c_spo2)
        bpm = float(c_bpm)

        query_point1 = np.array([age,heart_rate,temp,spo2,bpm]).reshape(1,-1)
        query_point1 = scaler.transform(query_point1)
        res = sv_classifier_health.predict(query_point1)
        res = res[0]
        res = res.capitalize()
        if(res=='Infected'):
            res = "Not Healthy"
        else:
            res = "Healthy"
        c_name = c_name.capitalize()
        c_gender = c_gender.capitalize()
        data = [{"Name : ": c_name, "Gender  :  " : c_gender , "Age  :  " : c_age,"Heart Rate  :  ": c_heart_rate,"Body Temperature  :  ":c_temp,"SpO2  :  ":c_spo2,"Beats per minute  :  ":c_bpm,"Health Status  :  ":res}]
        return render_template('first_page.html',result = data)
    
@app.route('/diagnosis',methods=['GET','POST'])
def diagnosis():
    if request.method == 'GET':
        return render_template('second_page.html')
    else : 
        c_name = request.form.get('name')
        c_age = request.form.get('age')
        c_gender = request.form.get('gender')
        c_symptom1 = request.form.get('symptom1')
        c_symptom2 = request.form.get('symptom2')
        c_symptom3 = request.form.get('symptom3')
        c_symptom4 = request.form.get('symptom4')
        c_symptom5 = request.form.get('symptom5')

        name = c_name.capitalize()
        gender = c_gender.capitalize()


        col = np.array(columns)
        symptoms = {}
        for i in col:
            symptoms[i] = 0
        # symptoms.popitem()
        symptom1 = "Symptom_0_"+c_symptom1
        symptom2 = "Symptom_1_"+c_symptom2
        symptom3 = "Symptom_2_"+c_symptom3
        symptom4 = "Symptom_3_"+c_symptom4
        symptom5 = "Symptom_4_"+c_symptom5
        query = np.array([symptom1,symptom2,symptom3,symptom4,symptom5])

        for i in query:
            symptoms[i] = 1


        q = np.array(list(symptoms.values())).reshape(1,-1)
        disease = nb_classifier.predict(q)
        disease = disease[0]

        df = precaution_data[precaution_data['Disease']==disease]
        precuation = np.array(df).reshape(1,-1)
        precuation = precuation[0][1:]

        description = description_data[description_data['Disease']==disease]
        description = np.array(description)
        description = description[0][1]

        doctors = doctor_data[doctor_data['name of disease']==disease]
        doctors = np.array(doctors)
        doctors = doctors[0][1]

        result = [{"Name : " : name , "Age : " : c_age , "Gender : " : gender,"Disease : " : disease ,"Description of the disease : " : description ,"Precautions : " : precuation,"To be suggested to meet this type of doctor : " : doctors}]
        
        return render_template('second_page.html',answer = result)



#####################################


if __name__ == '__main__':
    app.run(debug=True)
