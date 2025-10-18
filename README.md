# 🏥 Multiple Disease Prediction System  
### _Machine Learning Application for Health Diagnosis_  

---

## 🌐 Project Overview

This project is a **comprehensive disease prediction system** capable of predicting multiple critical diseases including **Breast Cancer, Diabetes, Heart Disease, and Parkinson’s Disease**.  
It leverages **machine learning models** with preprocessed medical datasets and provides an **interactive interface** via **Streamlit** for real-time predictions.

---

## 🎯 Objectives

- Predict the risk of **Diabetes**, **Heart Disease**, **Parkinson’s Disease**, and **Breast Cancer**.  
- Deploy trained models for **interactive health assessment**.  
- Standardize input data using pre-trained **scalers** for robust predictions.  
- Offer a **user-friendly web interface** for healthcare practitioners and individuals.

---

## ⚙️ Technical Stack


| Category             | Tools / Libraries                           |
|----------------------|---------------------------------------------|
| **Language**         |  ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) |
| **Data Processing**  | ![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white), ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)                                         |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)(SVM, Logistic Regression) |
| **Serialization**    | `pickle` |
| **Web Deployment**   |  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white), `streamlit_option_menu` |
| **UI / Interaction** | Streamlit sidebar & forms                    |

---

## 🧩 Pipeline Architecture

### **Workflow**
1. **Data Loading**  
   - Load CSV datasets for each disease.  
   - Load pre-trained models and scalers using `pickle`.  
2. **Preprocessing**  
   - Validate input fields and convert to numeric arrays.  
   - Standardize features using saved scalers.  
3. **Prediction**  
   - Apply corresponding ML model for each disease.  
   - Generate a clear prediction output (e.g., “Diabetic” / “Not Diabetic”).  
4. **Deployment**  
   - Interactive Streamlit interface with **sidebar navigation**.  
   - Separate pages for each disease prediction.

---

## 🧠 Implemented Models

| Disease                 | Features                                                            |Model                      |
|-------------------------|---------------------------------------------------------------------|---------------------------|
| **Diabetes**            | Glucose, BMI, Age, Blood Pressure, etc.                             | SVM / Logistic Regression |
| **Heart Disease**       | Age, Sex, Cholesterol, Blood Pressure, ECG readings.                | SVM / Logistic Regression |
| **Parkinson’s Disease** | Voice measurements (F0, jitter, shimmer, HNR, etc.)                 | Logistic Regression / SVM |
| **Breast Cancer**       | Tumor characteristics (radius, texture, perimeter, concavity, etc.) | SVM / Logistic Regression |

---

## 📊 Results Overview

- Models are pre-trained and achieve high accuracy on respective datasets.  
- Standardized input ensures **reliable predictions across all users**.  
- Interactive Streamlit interface provides **instant feedback** based on input data.

---
## 🧩 Repository Structure
```bash
📁 DISEASE PREDICTION SYSTEM/
│
├── 📊 Datasets/
│   ├── cancerData.csv              # Breast cancer patient data
│   ├── diabetes.csv                # Diabetes patient records
│   ├── heart_disease_data.csv      # Cardiovascular disease data
│   └── parkinsons_data.csv         # Parkinson's disease measurements
│
├── 🤖 model&Scalers/
│   ├── 📁 models/                  # Trained machine learning models
│   │   ├── breastCancer_model.sav
│   │   ├── diabetes_model.sav
│   │   ├── heart_model.sav
│   │   └── parkinsons_model.sav
│   │
│   └── 📁 Scalers/                 # Feature scaling objects
│       ├── breastCancer_scaler.sav
│       ├── Diabetes_scaler.sav
│       ├── heart_Scaler.sav
│       └── parkinsons_Scaler.sav
│
├── 📓 Jupyter Notebooks/           # Model development and training
│   ├── Breast Cancer Classification.ipynb
│   ├── Diabetes_Prediction.ipynb
│   ├── Heart Disease Prediction.ipynb
│   └── Parkinson's Disease_Prediction.ipynb
│
├── 🚀 Deployment/
│   └── multiple disease prediction.py    # Main Streamlit application
│
└── 📄 Documentation/
    └── README.md                    # Project documentation
```
---
## ⚠️ Usage
```bash
# 1. Clone the repository
git clone <repository_url>
cd DISEASE_PREDICTION_SYSTEM

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run src/multiple\ disease\ prediction.py
```
---

## 🧭 Navigate via the sidebar in the app
 
- **Diabetes Prediction**
- **Heart Disease Prediction**
- **Parkinson’s Disease Prediction**
- **reast Cancer Prediction**
---

## 👨‍💻 Author
**Ayoub Aamir**  

🎓 **Master Big Data & IoT**  
📍 *ENSAM Casablanca*  
📧 [aamir.ayoub@ensam-casa.ma](mailto:aamir.ayoub@ensam-casa.ma)

🔗 **Connect with me:**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayoub-aamir)  
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AamirAyoub123)
