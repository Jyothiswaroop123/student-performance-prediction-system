import joblib
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load ML model and encoders
model = joblib.load(BASE_DIR / "model" / "student_model.pkl")
encoders = joblib.load(BASE_DIR / "model" / "label_encoders.pkl")

import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.sidebar.title("🎓 Student Performance Prediction")

st.sidebar.markdown("---")

st.sidebar.info("""
Machine Learning Project

Algorithm:
Random Forest Classifier

Model Accuracy:
89.86%
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by:")
st.sidebar.write("Jyothiswaroop")

st.title("🎓 Student Performance Prediction System")

st.markdown("---")

st.header("Student Information")

col1, col2 = st.columns(2)

with col1:

    hours = st.slider("Hours Studied", 0, 30, 10)

    attendance = st.slider("Attendance (%)", 0, 100, 75)

    previous = st.slider("Previous Scores", 0, 100, 60)

    sleep = st.slider("Sleep Hours", 0, 12, 7)

    tutoring = st.slider("Tutoring Sessions", 0, 8, 2)

    physical = st.slider("Physical Activity", 0, 6, 3)

    parental = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

    resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

    motivation = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

    income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

with col2:

    internet = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )

    teacher = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )

    school = st.selectbox(
        "School Type",
        ["Public", "Private"]
    )

    peer = st.selectbox(
        "Peer Influence",
        ["Positive", "Neutral", "Negative"]
    )

    learning = st.selectbox(
        "Learning Disabilities",
        ["Yes", "No"]
    )

    education = st.selectbox(
        "Parental Education",
        ["High School", "College", "Postgraduate"]
    )

    distance = st.selectbox(
        "Distance from Home",
        ["Near", "Moderate", "Far"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )
import pandas as pd

input_data = pd.DataFrame([{
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Parental_Involvement": parental,
    "Access_to_Resources": resources,
    "Extracurricular_Activities": extracurricular,
    "Sleep_Hours": sleep,
    "Previous_Scores": previous,
    "Motivation_Level": motivation,
    "Internet_Access": internet,
    "Tutoring_Sessions": tutoring,
    "Family_Income": income,
    "Teacher_Quality": teacher,
    "School_Type": school,
    "Peer_Influence": peer,
    "Physical_Activity": physical,
    "Learning_Disabilities": learning,
    "Parental_Education_Level": education,
    "Distance_from_Home": distance,
    "Gender": gender
}])

# Encode categorical columns
for column, encoder in encoders.items():
    if column in input_data.columns:
        input_data[column] = encoder.transform(input_data[column])

if st.button("Predict Performance"):

    # ==========================================
    # PREDICT STUDENT PERFORMANCE
    # ==========================================

    prediction = model.predict(input_data)
    result = prediction[0]

    # Prediction probabilities
    prob = model.predict_proba(input_data)[0]

    # ==========================================
    # STUDENT DETAILS
    # ==========================================

    st.subheader("📝 Student Details")

    # Human-readable student information
    display_data = pd.DataFrame([{
        "Hours Studied": hours,
        "Attendance (%)": attendance,
        "Previous Scores": previous,
        "Sleep Hours": sleep,
        "Tutoring Sessions": tutoring,
        "Physical Activity": physical,
        "Parental Involvement": parental,
        "Access to Resources": resources,
        "Extracurricular Activities": extracurricular,
        "Motivation Level": motivation,
        "Internet Access": internet,
        "Family Income": income,
        "Teacher Quality": teacher,
        "School Type": school,
        "Peer Influence": peer,
        "Learning Disabilities": learning,
        "Parental Education": education,
        "Distance from Home": distance,
        "Gender": gender
    }])

    st.dataframe(
        display_data,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # PREDICTION CONFIDENCE
    # ==========================================

    st.subheader("📊 Prediction Confidence")

    st.progress(float(max(prob)))

    st.metric(
        "Prediction Confidence",
        f"{max(prob) * 100:.2f}%"
    )

    st.write(
        f"🟢 High : {prob[1] * 100:.2f}%"
    )

    st.write(
        f"🟡 Average : {prob[0] * 100:.2f}%"
    )

    st.write(
        f"🔴 Low : {prob[2] * 100:.2f}%"
    )

    st.markdown("---")

    # ==========================================
    # PREDICTED PERFORMANCE
    # ==========================================

    if result == 1:

        st.success(
            "🟢 Predicted Performance : HIGH"
        )

        st.subheader("📚 Recommendations")

        st.write(
            "✅ Continue your current study routine."
        )

        st.write(
            "✅ Maintain excellent attendance."
        )

        st.write(
            "✅ Participate in extracurricular activities."
        )

        st.write(
            "✅ Keep solving previous question papers."
        )

        performance_text = "HIGH"

    elif result == 0:

        st.warning(
            "🟡 Predicted Performance : AVERAGE"
        )

        st.subheader("📚 Recommendations")

        st.write(
            "✅ Increase study hours."
        )

        st.write(
            "✅ Improve class attendance."
        )

        st.write(
            "✅ Revise subjects every day."
        )

        st.write(
            "✅ Attend tutoring sessions regularly."
        )

        performance_text = "AVERAGE"

    else:

        st.error(
            "🔴 Predicted Performance : LOW"
        )

        st.subheader("📚 Recommendations")

        st.write(
            "✅ Study at least 2 extra hours every day."
        )

        st.write(
            "✅ Improve attendance."
        )

        st.write(
            "✅ Ask teachers for guidance."
        )

        st.write(
            "✅ Reduce distractions."
        )

        st.write(
            "✅ Practice mock tests regularly."
        )

        performance_text = "LOW"

    # ==========================================
    # DOWNLOAD REPORT
    # ==========================================

    report = f"""
STUDENT PERFORMANCE REPORT
================================

Student Information

Hours Studied: {hours}
Attendance: {attendance}%
Previous Scores: {previous}
Sleep Hours: {sleep}
Tutoring Sessions: {tutoring}
Physical Activity: {physical}

Parental Involvement: {parental}
Access to Resources: {resources}
Extracurricular Activities: {extracurricular}
Motivation Level: {motivation}
Internet Access: {internet}
Family Income: {income}
Teacher Quality: {teacher}
School Type: {school}
Peer Influence: {peer}
Learning Disabilities: {learning}
Parental Education: {education}
Distance from Home: {distance}
Gender: {gender}

================================
PREDICTION RESULT
================================

Predicted Performance: {performance_text}

Prediction Confidence

High: {prob[1] * 100:.2f}%
Average: {prob[0] * 100:.2f}%
Low: {prob[2] * 100:.2f}%

================================
Student Performance Prediction System
Developed using Python, Streamlit,
Pandas, Scikit-learn and Random Forest
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="Student_Performance_Report.txt",
        mime="text/plain"
    )