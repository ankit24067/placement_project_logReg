import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("🎓 Placement Predictor App")
st.markdown("Predict whether a student will get placed based on CGPA & IQ")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("placement.csv")
    df = df.iloc[:, 1:]
    return df

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df.iloc[:, 0:2]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train_scaled, y_train

model, scaler, X_train_scaled, y_train = train_model(df)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📥 Enter Student Details")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 6.0)
iq = st.sidebar.slider("IQ", 50, 200, 100)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Placement 🚀"):
    input_data = np.array([[cgpa, iq]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Student is likely to be PLACED")
    else:
        st.error("❌ Student is NOT likely to be placed")

# -------------------------------
# Visualization Section
# -------------------------------
st.subheader("📊 Dataset Visualization")

st.write("Scatter Plot of CGPA vs IQ")
st.scatter_chart(df, x="cgpa", y="iq", color="placement")

# -------------------------------
# Decision Boundary Plot
# -------------------------------
st.subheader("🧠 Decision Boundary (Model Understanding)")

fig = plt.figure()

plot_decision_regions(
    X_train_scaled,
    y_train.values,
    clf=model,
    legend=2
)

plt.xlabel("CGPA (scaled)")
plt.ylabel("IQ (scaled)")
plt.title("Decision Boundary - Logistic Regression")

st.pyplot(fig)

# -------------------------------
# Show Raw Data
# -------------------------------
if st.checkbox("Show Raw Data"):
    st.write(df.head())