
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/titanic_model.pkl")


# Feature engineering must match training
def create_features(data):
    data = data.copy()

    data["Title"] = (
        data["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    data["Title"] = data["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    common_titles = ["Mr", "Mrs", "Miss", "Master"]

    data["Title"] = data["Title"].where(
        data["Title"].isin(common_titles),
        "Rare"
    )

    data["FamilySize"] = (
        data["SibSp"] + data["Parch"] + 1
    )

    data["IsAlone"] = (
        data["FamilySize"] == 1
    ).astype(int)

    return data


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write(
    "Enter passenger details to get an ML prediction."
)

st.divider()

# Passenger details
st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Ticket Class",
        [1, 2, 3],
        index=2
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        value=8.05
    )

with col2:
    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )

st.divider()

# Predict
if st.button("Predict Survival", type="primary"):

    passenger = pd.DataFrame([{
        "PassengerId": 9999,
        "Pclass": pclass,
        "Name": "Example, Mr. Passenger",
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "TEST123",
        "Fare": fare,
        "Cabin": np.nan,
        "Embarked": embarked
    }])

    passenger = create_features(passenger)

    X_new = passenger.drop(
        columns=[
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ]
    )

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")

    st.metric(
        "Estimated survival probability",
        f"{probability:.1%}"
    )

    st.caption(
        "This is a statistical model prediction, "
        "not a guarantee of an actual outcome."
    )
