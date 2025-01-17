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
    
    #Gitting the inpur data from the user 
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')
        
    parkinsons_outcome=''
    
    input_data=[]
    
    #Create a button
    if st.button('Parkinsons Disease Test'):
        try:
            # Validate inputs (ensure no empty fields)
            inputs=[fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            if any(value.strip() == '' for value in inputs):
                st.error('All fields are required. Please fill in all the fields.')
            else:    
                # Convert inputs to floats and reshape for model
                input_data = np.array([float(value) for value in inputs]).reshape(1,-1)
                
                # Standardize the input data
                standardized_input_data = parkinsons_scaler.transform(input_data)
                
                # Make the prediction
                parkinsons_prediction = parkinsons_model.predict(standardized_input_data)
                
                if parkinsons_prediction[0] == 1:
                    parkinsons_outcome = "The person has Parkinson's disease"
                else:
                    parkinsons_outcome = "The person does not have Parkinson's disease"
           
        except ValueError:
             st.error("Please ensure all inputs are numeric.")
        except NameError:
            st.error("Input data processing error. Please try again.")            
    

    st.success(parkinsons_outcome) 

           
#Breast Cancer Prediction Page 
if(selected == 'Breast Cancer Prediction'):
    #Page Title 
    st.title("Breast Cancer Prediction")    
    
     
    #Gitting the inpur data from the user 
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        id = st.text_input('id')

    with col2:
        radius_mean = st.text_input('radius_mean')

    with col3:
        texture_mean = st.text_input('texture_mean')

    with col4:
        perimeter_mean = st.text_input('perimeter_mean')

    with col5:
        area_mean = st.text_input('area_mean')

    with col1:
        smoothness_mean = st.text_input('smoothness_mean')

    with col2:
        compactness_mean = st.text_input('compactness_mean')

    with col3:
        concavity_mean = st.text_input('concavity_mean')

    with col4:
        concave_points_mean = st.text_input('concave points_mean')
    
    with col5:
        symmetry_mean	 = st.text_input('symmetry_mean')    
    
    with col1:
        fractal_dimension_mean	 = st.text_input('fractal_dimension_mean')

    with col2:
        radius_se = st.text_input('radius_se')

    with col3:
        texture_se = st.text_input('texture_se')

    with col4:
        perimeter_se = st.text_input('perimeter_se')

    with col5:
        area_se = st.text_input('area_se')

    with col1:
        smoothness_se = st.text_input('smoothness_se')

    with col2:
        compactness_se = st.text_input('compactness_se')

    with col3:
        concavity_se = st.text_input('concavity_se')

    with col4:
        concave_points_se = st.text_input('concave points_se')

    with col5:
        symmetry_se = st.text_input('symmetry_se')

    with col1:
        fractal_dimension_se = st.text_input('fractal_dimension_se')

    with col2:
        radius_worst = st.text_input('radius_worst')

    with col3:
        texture_worst = st.text_input('texture_worst')
   
    with col4:
        perimeter_worst = st.text_input('perimeter_worst')

    with col5:
        area_worst = st.text_input('area_worst')

    with col1:
        smoothness_worst = st.text_input('smoothness_worst')
    
    with col2:
        compactness_worst = st.text_input('compactness_worst')

    with col3:
        concavity_worst = st.text_input('concavity_worst')
   
    with col4:
        concave_points_worst = st.text_input('concave points_worst')

    with col5:
        symmetry_worst = st.text_input('symmetry_worst')

    with col1:
        fractal_dimension_worst = st.text_input('fractal_dimension_worst')     
    
    # Map the inputs to their respective dataset column names
    data = {
        'id': id,
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,  # Map with the original column name
        'symmetry_mean':symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,  # Map with the original column name
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,  # Map with the original column name
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst,
        }     
    
    breastCancer_outcome=''
    
    input_data=[]
    
    #Create a button
    if st.button('Breast Cancer Test'):
        try:
            # Validate inputs (ensure no empty fields)
            inputs=[id,radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                    compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,
                    radius_se, texture_se, perimeter_se,area_se, smoothness_se, compactness_se,
                    concavity_se, concave_points_se, symmetry_se,fractal_dimension_se,
                    radius_worst, texture_worst, perimeter_worst,area_worst,smoothness_worst,compactness_worst,
                   concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
            
            
            if any(value.strip() == '' for value in inputs):
                st.error('All fields are required. Please fill in all the fields.')
            else:    
                # Convert inputs to floats and reshape for model
                input_data = np.array([float(value) for value in inputs]).reshape(1,-1)
                
                # Standardize the input data
                standardized_input_data = breastCancer_scaler.transform(input_data)
                
                # Make the prediction
                breastCancer_prediction =breastCancer_model.predict(standardized_input_data)
                
                if breastCancer_prediction[0] == 1:
                    breastCancer_prediction = "The Breast Cancer is Benign"
                else:
                    breastCancer_prediction = "The Breast cancer is Malignant"
           
        except ValueError:
             st.error("Please ensure all inputs are numeric.")
        except NameError:
            st.error("Input data processing error. Please try again.")            
    

    st.success(breastCancer_outcome)
    
    