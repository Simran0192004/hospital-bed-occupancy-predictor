import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('hospital_data.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['day_of_week'] = df['date'].dt.dayofweek

columns_needed = [
    'previous_day_admission_adult_covid_confirmed',
    'staffed_adult_icu_bed_occupancy',
    'total_adult_patients_hospitalized_confirmed_covid',
    'inpatient_beds_used',
    'inpatient_beds_used_covid'
]
df = df.dropna(subset=columns_needed)

# Select features and target
features = ['previous_day_admission_adult_covid_confirmed',
            'staffed_adult_icu_bed_occupancy',
            'day_of_week']
target = 'inpatient_beds_used_covid'

X = df[features]
y = df[target]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏥 Hospital Bed Occupancy Predictor")
st.write("Enter data to predict COVID inpatient bed usage:")

input_data = {}
for col in features:
    input_data[col] = st.number_input(col.replace('_', ' ').capitalize(), value=0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Inpatient Beds Used for COVID: **{int(prediction)}**")

# Optional: Show the scatter plot
if st.checkbox("Show Actual vs Predicted Plot"):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, c='skyblue', edgecolors='k', alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Hospital Bed Occupancy")
    st.pyplot(fig)