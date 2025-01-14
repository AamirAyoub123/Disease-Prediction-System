# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:15:11 2025

@author: ayoub
"""

import pickle 
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu 


st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Loading the saved model files

diabetes_model = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/models/heart_model.sav', 'rb'))

breastCancer_model = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/models/breastCancer_model.sav', 'rb'))

parkinsons_model = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/models/parkinsons_model.sav', 'rb'))


# Loading the saved scaler files

diabetes_scaler = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/Scalers/Diabetes_scaler.sav', 'rb'))

heart_scaler = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/Scalers/heart_Scaler.sav', 'rb'))

breastCancer_scaler = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/Scalers/breastCancer_scaler.sav', 'rb'))

parkinsons_scaler = pickle.load(open('D:/machinlearning/Disease Prediction System/model&Scalers/Scalers/parkinsons_Scaler.sav', 'rb'))



#Sidebar for navigate 

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                           
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction'],
                           
                            menu_icon='hospital-fill',
                           
                            icons=['activity','heart','person-walking','bandaid'],
                            
                            default_index = 0)



#Diabetes Prediction Page 
if(selected == 'Diabetes Prediction'):
    #Page Title 
    st.title("Diabetes Prediction")
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        Pregnancies=st.text_input('Number of Pregnancies')
    with col2:    
        Glucose = st.text_input('Glucose Level')
    with col3:    
        BloodPressure = st.text_input('Blood Pressure level')
    with col1:    
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:  
        Insulin = st.text_input('Insulin Level')
    with col3:     
        BMI = st.text_input('BMI value')
    with col1:    
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function level')
    with col2:    
        Age = st.text_input('Age of the Person')
    
    
    #Code for Prediction
    diab_outcome = ''
    

    #Creating a button
    if st.button('Diabetes Test Result'):
       try:
            # Validate inputs (ensure no empty fields)
            inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

            if any(value.strip() == '' for value in inputs):
                st.error("All fields are required. Please fill in all the fields.")
            else:
                # Convert inputs to floats and reshape for model
                input_data = np.array([float(value) for value in inputs]).reshape(1, -1)

                # Standardize the input data
                standardized_input_data = diabetes_scaler.transform(input_data)

                # Make the prediction
                diab_prediction = diabetes_model.predict(standardized_input_data)

                if diab_prediction[0] == 1:
                    diab_outcome = 'The person is Diabetic'
                else:
                    diab_outcome = 'The person is not Diabetic'

       except ValueError:
            st.error("Please ensure all inputs are numeric.")
       except NameError:
            st.error("Input data processing error. Please try again.")

    st.success(diab_outcome)  
    

    
    
#Heart Disease Prediction Page 
if(selected == 'Heart Disease Prediction'):
    #Page Title 
    st.title("Heart Disease Prediction")
    
    
    #Gitting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        age = st.text_input('Age')
    
    with col2:
        sex = st.text_input('Sex')
 
    with col3:
        cp = st.text_input('Chest Pain Types')
     
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:                                                   
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    
    #Code for Prediction
    heart_outcome = ''
    
    input_data=[]
    #Create a button 
    if st.button('Heart Disease Test'):
       try : 
                # Validate inputs (ensure no empty fields)
                inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

                if any(value.strip() == '' for value in inputs):
                    st.error("All fields are required. Please fill in all the fields.")
                else:
                    # Convert inputs to floats and reshape for model
                    input_data = np.array([float(value) for value in inputs]).reshape(1, -1)
              
                    # Standardize the input data
                    standardized_input_data = heart_scaler.transform(input_data)

                    # Make the prediction
                    heart_prediction  = heart_disease_model.predict(standardized_input_data)

                    if heart_prediction [0] == 1:
                        heart_outcome = 'The person is having heart disease'
                    else:
                        heart_outcome = 'The person does not have any heart disease'

       except ValueError:
            st.error("Please ensure all inputs are numeric.")
       except NameError:
           st.error("Input data processing error. Please try again.")

    st.success(heart_outcome)  


#Parkinsons Prediction Page 
if(selected == 'Parkinsons Prediction'):
    #Page Title 
    st.title("Parkinsons Prediction")    


#Breast Cancer Prediction Page 
if(selected == 'Breast Cancer Prediction'):
    #Page Title 
    st.title("Breast Cancer Prediction")    