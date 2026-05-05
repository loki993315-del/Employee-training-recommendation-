# =============================================================================
# FINAL PROJECT (UPDATED): Training Recommendation System
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Smart Training System", page_icon="🎓")

FILE = "employee_training_with_label.xlsx"

# ---------------------------------------------------------------------
# DATASET GENERATION
# ---------------------------------------------------------------------
def generate_dataset(path, n=1500):

    np.random.seed(42)
    roles = ["Data Analyst", "Backend Developer", "Frontend Developer"]

    data = []

    for i in range(n):
        role = np.random.choice(roles)
        exp = np.random.randint(0, 10)
        test_score = np.random.randint(30, 100)
        project_score = np.random.randint(30, 100)

        final_score = (test_score * 0.4) + (project_score * 0.6)

        if final_score < 50:
            label = "Beginner"
        elif final_score < 70:
            label = "Intermediate"
        else:
            label = "Advanced"

        data.append([
            f"EMP{i+1:04d}",
            role,
            exp,
            test_score,
            project_score,
            final_score,
            label
        ])

    df = pd.DataFrame(data, columns=[
        "employee_id","role","experience",
        "test_score","project_score",
        "final_score","label"
    ])

    df.to_excel(path, index=False)

# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():

    if not os.path.exists(FILE):
        generate_dataset(FILE)

    df = pd.read_excel(FILE)

    X = df[["experience","test_score","project_score"]]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X_scaled,y,test_size=0.2,random_state=42
    )

    dt_model = DecisionTreeClassifier(max_depth=5)
    dt_model.fit(X_train,y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

    rf_model = RandomForestClassifier(n_estimators=200)
    rf_model.fit(X_train,y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    return df, dt_model, rf_model, scaler, dt_acc, rf_acc

# ---------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------
def main():

    df, dt_model, rf_model, scaler, dt_acc, rf_acc = load_models()

    st.title("🎓 Intelligent Training Recommendation System")

    st.write("Decision Tree Accuracy:", round(dt_acc*100,2), "%")
    st.write("Random Forest Accuracy:", round(rf_acc*100,2), "%")

    st.write("---")

    emp_id = st.text_input("Enter Employee ID (Example: EMP0001)")
    algo = st.selectbox("Select Algorithm", ["Decision Tree","Random Forest"])

    if st.button("Analyze Employee"):

        emp_data = df[df["employee_id"] == emp_id]

        if not emp_data.empty:

            role = emp_data["role"].values[0]
            experience = int(emp_data["experience"].values[0])
            test_score = int(emp_data["test_score"].values[0])
            project_score = int(emp_data["project_score"].values[0])

            st.success("Employee Found ✅")

            st.write("👤 Role:", role)
            st.write("📊 Experience:", experience)
            st.write("📝 Test Score:", test_score)
            st.write("💻 Project Score:", project_score)

            final_score = (test_score * 0.4) + (project_score * 0.6)
            st.write("⭐ Final Score:", round(final_score,2))

            # Model Prediction
            X_input = np.array([[experience,test_score,project_score]])
            X_input = scaler.transform(X_input)

            if algo == "Decision Tree":
                prediction = dt_model.predict(X_input)[0]
            else:
                prediction = rf_model.predict(X_input)[0]

            # ---------------- DOMAIN RECOMMENDATION ----------------
            if role == "Data Analyst":
                skills = {"Python": test_score, "SQL": project_score}

            elif role == "Backend Developer":
                skills = {"Java": test_score, "SQL": project_score}

            elif role == "Frontend Developer":
                skills = {"Web Development": test_score}

            weak_skill = min(skills, key=skills.get)
            strong_skill = max(skills, key=skills.get)

            # Experience-based level
            if experience <= 2:
                level = "Basic"
            elif experience <= 5:
                level = "Intermediate"
            else:
                level = "Advanced"

            recommend_map = {
                "Python": f"{level} Python Training",
                "SQL": f"{level} SQL Training",
                "Java": f"{level} Java Training",
                "Web Development": f"{level} Web Development Training"
            }

            st.subheader("🎯 Recommended Training")
            st.success(recommend_map[weak_skill])

            # ---------------- STRONG SKILL ----------------
            st.subheader("💪 Strong Skill")
            st.success(strong_skill)

            # ---------------- CAREER GROWTH ----------------
            st.subheader("🚀 Career Growth")

            growth_map = {
                "Data Analyst": "Senior Data Analyst / Data Scientist",
                "Backend Developer": "Senior Backend Developer / System Architect",
                "Frontend Developer": "Senior Frontend Developer / UI Architect"
            }

            st.write("Current Role:", role)
            st.success(growth_map[role])

        else:
            st.error("Employee ID not found ❌")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()