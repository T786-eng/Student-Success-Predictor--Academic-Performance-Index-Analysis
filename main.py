import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD AND PREPROCESS DATA
# ---------------------------
# Load the dataset
df = pd.read_csv('python_learning_exam_performance.csv')

# Handle missing values in 'prior_programming_experience'
# Some rows have NaN or "None", we'll treat them as a single category
df['prior_programming_experience'] = df['prior_programming_experience'].fillna('None')

# Drop student_id as it doesn't help in prediction
df_model = df.drop('student_id', axis=1)

# Encode categorical variables ('country' and 'prior_programming_experience')
le = LabelEncoder()
df_model['country'] = le.fit_transform(df_model['country'])
df_model['prior_programming_experience'] = le.fit_transform(df_model['prior_programming_experience'])

# 2. FEATURE SELECTION & SPLITTING
# --------------------------------
# Target: passed_exam
# Features: all columns except passed_exam and final_exam_score (to avoid data leakage)
X = df_model.drop(['passed_exam', 'final_exam_score'], axis=1)
y = df_model['passed_exam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL TRAINING (Random Forest)
# ---------------------------------
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. EVALUATION
# -------------
y_pred = rf_model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. VISUALIZATIONS
# -----------------
# Feature Importance Plot
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
plt.title('Key Factors Predicting Exam Success')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_model.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()