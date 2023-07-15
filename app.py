from flask import Flask, request, render_template, redirect
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__, static_folder='assets')
model = pickle.load(open("cat_try.pkl", "rb"))


@app.route("/")
@cross_origin()

def home():
    return render_template("index.html")
    


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    # return render_template("predict.html")

    if request.method == "POST":
        
        #Age
        age_input=request.form["age"] 
        age = float(age_input)
        # print(age)        
        
        #BMI
        bmi_input=request.form["bmi"] 
        bmi = float(bmi_input)
        
        
        #BloodGlucose
        avg_glucose_level_input=request.form["avg_glucose_level"] 
        avg_glucose_level = float(avg_glucose_level_input) 
        
        #Ever Married
        ever_married = request.form.get('evermarried')
        if (ever_married == 'Yes'):
            ever_married_Yes = 1
            ever_married_No = 0
            
        elif (ever_married == 'No'):
            ever_married_Yes = 0
            ever_married_No = 1
        
        #Hypertension
        hypertension_input = request.form.get('hypertension')
        if (hypertension_input == 'Yes'):
            hypertension = 1            
            
        elif (hypertension_input == 'No'):
            hypertension = 0
            
        #heart_disease
        heart_disease_input = request.form.get('heartdisease')
        if (heart_disease_input == 'Yes'):
            heart_disease = 1            
            
        elif (heart_disease_input == 'No'):
            heart_disease = 0
            
        
        #Gender       
        Gender = request.form["Gender"]
        if (Gender == 'Male'):
            gender_Male = 1
            gender_Female = 0

        elif (Gender == 'Female'):
            gender_Male = 0
            gender_Female = 1
            
            
        #Residence Type       
        Residence_type = request.form["Residence_type"]
        if (Residence_type == 'Urban'):
            Residence_type_Urban = 1
            Residence_type_Rural = 0

        elif (Residence_type == 'Rural'):
            Residence_type_Urban = 0
            Residence_type_Rural = 1
            
            
        #Smoking Status       
        smoking_status = request.form["smoking_status"]
        if (smoking_status == 'smokes'):
            smoking_status_smokes = 1
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_Unknown = 0

        elif (smoking_status == 'never smoked'):
            smoking_status_smokes = 0
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_Unknown = 0
            
        elif (smoking_status == 'formerly Smoked'):
            smoking_status_smokes = 0
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_Unknown = 0
            
        elif (smoking_status == 'Unknown'):
            smoking_status_smokes = 0
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_Unknown = 1
            
            
        #Work Type       
        work_type = request.form["work_type"]
        if (work_type == 'Govt_job'):
            work_type_Govt_job = 1
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0

        elif (work_type == 'Private'):
            work_type_Govt_job = 0
            work_type_Never_worked = 0
            work_type_Private = 1
            work_type_Self_employed = 0
            work_type_children = 0
            
        elif (work_type == 'Self-employed'):
            work_type_Govt_job = 0
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
            
        elif (work_type == 'children'):
            work_type_Govt_job = 0
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 1
            
        elif (work_type == 'Never_worked'):
            work_type_Govt_job = 0
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
            
        df=[[age,
            bmi,
            avg_glucose_level,
            ever_married_Yes,
            ever_married_No,
            hypertension,
            heart_disease,
            gender_Male,
            gender_Female,
            Residence_type_Urban,
            Residence_type_Rural,
            smoking_status_smokes,
            smoking_status_formerly_smoked,
            smoking_status_never_smoked,
            smoking_status_Unknown,
            work_type_Govt_job,
            work_type_Never_worked,
            work_type_Private,
            work_type_Self_employed,
            work_type_children]]   
         
        df = pd.DataFrame(df, columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self_employed', 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown', 'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes'])
        prediction = model.predict(df)   
            
        # prediction=model.predict([[
        #     age,
        #     bmi,
        #     avg_glucose_level,
        #     ever_married_Yes,
        #     ever_married_No,
        #     hypertension,
        #     heart_disease,
        #     gender_Male,
        #     gender_Female,
        #     Residence_type_Urban,
        #     Residence_type_Rural,
        #     smoking_status_smokes,
        #     smoking_status_formerly_smoked,
        #     smoking_status_never_smoked,
        #     smoking_status_Unknown,
        #     work_type_Govt_job,
        #     work_type_Never_worked,
        #     work_type_Private,
        #     work_type_Self_employed,
        #     work_type_children
        # ]])
        
        
        
        
        if (prediction == 1):
            output = "Stroke"
        elif (prediction == 0):
            output = "No Stroke"
        
        # output=[age,bmi,avg_glucose_level,gender_Female,gender_Male,work_type_children,smoking_status_never_smoked,hypertension,heart_disease]
        
        return render_template('predict.html',prediction_text='Result is {}'.format(output))


    return render_template("predict.html")     
        
if __name__ == "__main__":
    app.run(debug=True)