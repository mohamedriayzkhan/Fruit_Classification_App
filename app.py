import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit Classification App",
    page_icon="🍎",
    layout="centered"
)

st.title("🍉 Fruit Name Prediction (Dynamic)")
st.write("Enter fruit characteristics to predict the fruit name")

# ---------------- LOAD MODEL ----------------
with open("model/random_forest_model.pkl", "rb") as f:
    model, feature_columns = pickle.load(f)

# ---------------- LOAD DATA (for dynamic UI) ----------------
df = pd.read_csv("data/dataset.csv")

# Identify feature types
target_col = "fruit_name"
categorical_cols = df.select_dtypes(include=["object"]).drop(columns=[target_col]).columns
numerical_cols = df.select_dtypes(exclude=["object"]).columns

st.subheader("🔧 Input Features")

user_input = {}

# ---------------- NUMERICAL INPUTS (DYNAMIC) ----------------
for col in numerical_cols:
    user_input[col] = st.number_input(
        label=col,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean())
    )

# ---------------- CATEGORICAL INPUTS (DYNAMIC) ----------------
for col in categorical_cols:
    user_input[col] = st.selectbox(
        label=col,
        options=sorted(df[col].unique())
    )

# ---------------- PREDICTION ----------------
if st.button("Predict Fruit 🍓"):
    input_df = pd.DataFrame([user_input])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"✅ Predicted Fruit: **{prediction}**")
