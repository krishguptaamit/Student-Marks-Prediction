
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load real student dataset
data = pd.read_csv("student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Total student records:", len(data))

# Select input features
X = data[["studytime", "failures", "absences", "G1", "G2"]]

# Final marks
y = data["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Training Completed!")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Save trained model
joblib.dump(model, "student_model.pkl")

print("\nModel saved successfully!")
