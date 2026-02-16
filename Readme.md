# Student Python Performance Predictor 🎓🐍

This project uses Machine Learning to analyze and predict student success in a Python programming course. Using a dataset of 3,000 students, the model identifies key behavioral patterns—like study hours and project completion—that lead to passing the final exam.

## 🚀 Overview
The goal of this project is to provide insights into student learning habits and predict whether a student will pass (`1`) or fail (`0`) the exam using a **Random Forest Classifier**.

### Key Features
* **Data Cleaning**: Handled missing values in programming experience.
* **Feature Engineering**: Encoded categorical data (Country, Experience) for model compatibility.
* **Predictive Modeling**: Achieved **~90% accuracy** in predicting exam outcomes.
* **Data Visualization**: Includes correlation heatmaps and feature importance charts.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **IDE:** VS Code

## 📊 Results & Insights
Based on the model's feature importance analysis:
1. **Prior Experience:** The strongest predictor of success.
2. **Learning Hours:** Weekly consistency is more important than "cramming."
3. **Project Work:** Hands-on project completion significantly boosts passing probability.



## 📂 Project Structure
```text
├── python_learning_exam_performance.csv  # Dataset
├── main.py                               # Model training & evaluation script
├── requirements.txt                      # Dependencies
└── README.md                             # Project documentation

