
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

print("===== STUDENT MARKS PREDICTION =====")

studytime = float(input("Weekly study time (1-4): "))
failures = int(input("Past class failures (0-3): "))
absences = int(input("Number of absences: "))
g1 = float(input("First period marks (0-20): "))
g2 = float(input("Second period marks (0-20): "))

# Prepare input
student = pd.DataFrame([{
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "G1": g1,
    "G2": g2
}])

# Predict final marks
prediction = model.predict(student)[0]
prediction = max(0, min(20, prediction))

print("\n===== PREDICTION RESULT =====")
print("Predicted Final Marks:", round(prediction, 2), "/ 20")
