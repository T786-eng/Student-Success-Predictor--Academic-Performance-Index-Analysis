import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Load data with Pandas
    try:
        df = pd.read_csv('Student_Performance.csv')
        print("Dataset loaded successfully.")

        # 2. Preprocessing
        # Convert 'Extracurricular Activities' (Yes/No) to binary (1/0)
        df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

        # 3. Define Features (X) and Target (y)
        X = df.drop('Performance Index', axis=1)
        y = df['Performance Index']

        # 4. Split data using Scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Initialize and train the Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6. Make Predictions
        y_pred = model.predict(X_test)

        # 7. Evaluate Model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared Score: {r2:.4f}")

        # 8. Generate and Auto-Save Graph
        # Plotting Actual vs Predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Actual vs Predicted')
        
        # Draw the 'Perfect Prediction' line
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, label='Perfect Fit')
        
        plt.xlabel('Actual Performance Index')
        plt.ylabel('Predicted Performance Index')
        plt.title('Student Performance: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        
        # Automatic Save
        graph_name = "student_performance_graph.png"
        plt.savefig(graph_name)
        print(f"Graph saved automatically as: {graph_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()