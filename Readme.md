# 🎓 Student Performance Predictor

This project is part of my **30 Days of Project Building** series. It uses Machine Learning to predict a student's **Performance Index** based on academic and lifestyle habits.

## 📝 Project Overview
The goal of this project is to understand how different factors—like study hours, sleep, and previous scores—impact overall academic achievement. By using a **Linear Regression** model, we can quantify these relationships and predict future performance.

## 🚀 Features
* **Data Cleaning:** Automatically maps categorical data (Extracurricular Activities) to numerical values.
* **Predictive Analysis:** Uses `Scikit-learn` to train a Linear Regression model.
* **Accuracy Metrics:** Outputs Mean Squared Error (MSE) and R-squared (R2) scores to the console.
* **Visual Insights:** Generates a scatter plot comparing Actual vs. Predicted values with a "Perfect Fit" line.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
* **Environment:** VS Code

## 📊 Dataset Attributes
The model is trained on the following features:
1. **Hours Studied**: Total time spent studying.
2. **Previous Scores**: Marks obtained in previous exams.
3. **Extracurricular Activities**: Participation in outside activities (Yes/No).
4. **Sleep Hours**: Average daily sleep duration.
5. **Sample Papers**: Number of practice papers completed.
6. **Target**: Performance Index (0-100).

## ⚙️ How to Run
1. Ensure you have `Student_Performance.csv` in the same folder as `main.py`.
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
