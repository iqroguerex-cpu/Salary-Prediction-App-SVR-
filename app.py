import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SVR Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# -----------------------------
# Title Section
# -----------------------------
st.title("💰 Salary Prediction App")
st.subheader("Support Vector Regression (SVR) Model")

st.markdown(
"""
This app predicts the **salary of an employee based on their position level**  
using a **Support Vector Regression (SVR)** machine learning model.

Adjust the **position level** using the slider to see the predicted salary.
"""
)

st.divider()


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Position_Salaries.csv")


dataset = load_data()

# Show dataset
with st.expander("📊 View Dataset"):
    st.dataframe(dataset)


# -----------------------------
# Prepare Data
# -----------------------------
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# -----------------------------
# Train Model
# -----------------------------
regressor = SVR(kernel="rbf")
regressor.fit(X, y)


# -----------------------------
# User Input Section
# -----------------------------
st.header("🔢 Choose Position Level")
st.info("You can enter decimal levels (e.g., 6.5) to estimate salaries between positions.")

level = st.slider(
    "Position Level",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

st.write(f"Selected Position Level: **{level}**")


# -----------------------------
# Prediction
# -----------------------------
prediction = sc_y.inverse_transform(
    regressor.predict(sc_X.transform([[level]])).reshape(-1, 1)
)

st.success(f"💵 Predicted Salary: **${prediction[0][0]:,.2f}**")


st.divider()


# -----------------------------
# Visualization Section
# -----------------------------
st.header("📈 Model Visualization")

st.markdown(
"""
- 🔴 **Red dots** represent actual salaries from the dataset  
- 🔵 **Blue curve** represents the SVR model predictions
"""
)

# Create smooth curve
X_original = sc_X.inverse_transform(X)

X_grid = np.arange(
    np.min(X_original),
    np.max(X_original),
    0.1
)

X_grid = X_grid.reshape((len(X_grid), 1))

# Plot graph
fig, ax = plt.subplots()

ax.scatter(
    sc_X.inverse_transform(X),
    sc_y.inverse_transform(y),
    color="red",
    label="Actual Salary"
)

ax.plot(
    X_grid,
    sc_y.inverse_transform(
        regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)
    ),
    color="blue",
    label="SVR Prediction"
)

ax.set_title("Salary Prediction using SVR")
ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")

ax.legend()

st.pyplot(fig)


st.divider()


# -----------------------------
# About Section
# -----------------------------
st.header("ℹ️ About the Model")

st.markdown(
"""
This project uses **Support Vector Regression (SVR)**.

### How it works

1️⃣ The dataset contains **position levels and salaries**  
2️⃣ Data is **scaled using StandardScaler**  
3️⃣ The SVR model learns the relationship between level and salary  
4️⃣ The model predicts salary for any new position level  

### Why SVR?

SVR works well when:

- Relationships are **non-linear**
- Dataset is **small**
- We want a **smooth prediction curve**

Kernel used: **RBF (Radial Basis Function)**
"""
)


# Footer
st.caption("Built with ❤️ using Streamlit and Scikit-Learn by Chinmay")
