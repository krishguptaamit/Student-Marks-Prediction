
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("student_model.pkl")

# Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Prepare features and target
X = data[["studytime", "failures", "absences", "G1", "G2"]]
y = data["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Predictions for evaluation
test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0b1120;
    color: #ffffff;
}

/* All normal text */
.stApp p,
.stApp label,
.stApp span,
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #ffffff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Header */
header[data-testid="stHeader"] {
    background-color: #0b1120;
}

/* Text input */
.stTextInput input {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* Number input */
.stNumberInput input {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #475569 !important;
}

/* Select box */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1f2937 !important;
    color: #ffffff !important;
}

/* Dropdown selected text */
.stSelectbox div[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* Prediction button */
div.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 45px;
    font-weight: bold;
}

div.stButton > button p {
    color: #ffffff !important;
}

/* Download button */
div[data-testid="stDownloadButton"] button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 45px;
    font-weight: bold;
}

div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span {
    color: #ffffff !important;
}

/* Info and warning messages */
div[data-testid="stAlert"] {
    background-color: #172554 !important;
    border: 1px solid #2563eb !important;
    border-radius: 10px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: #ffffff !important;
}

/* Result cards */
.result-box {
    background-color: #172554;
    padding: 22px;
    border-radius: 12px;
    text-align: center;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎓 Student Prediction")
st.sidebar.write("Machine Learning Project")
st.sidebar.divider()

st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Dataset: UCI Student Performance")
st.sidebar.write("Records: 395")
st.sidebar.write("Prediction Scale: 0–20")

# ---------------- MAIN HEADING ----------------

st.title("🎓 Student Marks Prediction System")

st.write(
    "Predict a student's final marks using Machine Learning."
)

st.divider()

# ---------------- STUDENT DETAILS ----------------

st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    student_name = st.text_input("Student Name")

    roll_number = st.text_input("Roll Number")

    studytime = st.selectbox(
        "Weekly Study Time",
        [1, 2, 3, 4],
        index=2,
        format_func=lambda x: {
            1: "1 - Less than 2 hours",
            2: "2 - 2 to 5 hours",
            3: "3 - 5 to 10 hours",
            4: "4 - More than 10 hours"
        }[x]
    )

    failures = st.number_input(
        "Past Class Failures",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )

with col2:

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )

    g1 = st.number_input(
        "First Period Marks (G1)",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.5
    )

    g2 = st.number_input(
        "Second Period Marks (G2)",
        min_value=0.0,
        max_value=20.0,
        value=14.0,
        step=0.5
    )

# ---------------- PREDICT BUTTON ----------------

st.write("")

predict_button = st.button(
    "🔮 Predict Final Marks",
    use_container_width=True
)

if predict_button:

    if not student_name.strip() or not roll_number.strip():

        st.warning(
            "Please enter Student Name and Roll Number."
        )

    else:

        input_data = pd.DataFrame([{
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": g1,
            "G2": g2
        }])

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(20, prediction))
        prediction = round(prediction, 2)

        if prediction >= 16:
            performance = "Excellent"

        elif prediction >= 12:
            performance = "Good"

        elif prediction >= 10:
            performance = "Average"

        else:
            performance = "Needs Improvement"

        st.divider()
        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown(
                f"""
                <div class="result-box">
                    <h3>Predicted Final Marks</h3>
                    <h1>{prediction} / 20</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.markdown(
                f"""
                <div class="result-box">
                    <h3>Performance</h3>
                    <h1>{performance}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Add to history
        st.session_state.history.append({
            "Student Name": student_name,
            "Roll Number": roll_number,
            "Study Time": studytime,
            "Failures": failures,
            "Absences": absences,
            "G1": g1,
            "G2": g2,
            "Predicted Marks": prediction,
            "Performance": performance
        })

# ---------------- MODEL PERFORMANCE ----------------

st.divider()
st.subheader("📈 Model Performance Analysis")

metric1, metric2 = st.columns(2)

metric1.metric(
    "Mean Absolute Error (MAE)",
    round(mae, 2)
)

metric2.metric(
    "R² Score",
    round(r2, 2)
)

# Graph
fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(
    y_test,
    test_predictions,
    alpha=0.7,
    label="Predicted Marks"
)

ax.plot(
    [0, 20],
    [0, 20],
    linestyle="--",
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Marks")
ax.set_ylabel("Predicted Marks")
ax.set_title("Actual vs Predicted Student Marks")
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# ---------------- PREDICTION HISTORY ----------------

st.divider()
st.subheader("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Student Report (CSV)",
        data=csv,
        file_name="student_marks_report.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No predictions yet. Enter student details and click Predict."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Student Marks Prediction System | Python • Streamlit • Scikit-learn"
)
