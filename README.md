# 🎓 Student Performance Prediction System

## 📌 Project Overview

The Student Performance Prediction System is a machine learning application designed to analyze student academic and personal attributes and predict their performance level as High, Average, or Low.

The system uses a Random Forest Classifier and provides predictions through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze student performance data
- Identify factors related to academic performance
- Predict student performance levels
- Compare multiple machine learning algorithms
- Provide prediction confidence
- Generate personalized recommendations
- Present predictions through an interactive dashboard

## 📊 Dataset

Dataset contains 6,607 student records with 20 columns, including:

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Extracurricular Activities
- Sleep Hours
- Previous Scores
- Motivation Level
- Internet Access
- Tutoring Sessions
- Family Income
- Teacher Quality
- School Type
- Peer Influence
- Physical Activity
- Learning Disabilities
- Parental Education Level
- Distance from Home
- Gender
- Exam Score

## 🤖 Machine Learning Models

The following models were evaluated:

| Model | Accuracy |
|---|---:|
| Decision Tree | 84.64% |
| Random Forest | 89.86% |
| Logistic Regression | 89.49% |
| KNN | 86.01% |

### Best Model

Random Forest Classifier

**Accuracy: 89.86%**

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
Student Performance Prediction System/
│
├── dashboard/
│   └── app.py
│
├── model/
│   ├── student_model.pkl
│   └── label_encoders.pkl
│
├── dataset/
│   ├── student_data.csv
│   └── student_data_cleaned.csv
│
├── notebook/
│   └── StudentPerformancePrediction.ipynb
│
├── requirements.txt
└── README.md