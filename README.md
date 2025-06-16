# 🏥 Hospital Bed Occupancy Predictor

A machine learning project that predicts the number of COVID-related hospital bed occupancies using real-time healthcare data. Built with a Linear Regression model to assist in effective hospital capacity planning.

---

## 🚀 Features

- Predicts COVID inpatient bed usage using ICU and admission data
- Utilizes real-world hospital data for reliable forecasting
- Clean and modular code structure for easy customization
- Includes visualization of actual vs predicted values

---

## 📁 Dataset

This project uses the publicly available dataset from [HealthData.gov](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u), which contains daily reports on hospital capacity and COVID-19 impacts across the U.S.

---

## 🧠 Machine Learning Model

- **Algorithm**: Linear Regression  
- **Features Used**:
  - Previous Day's Adult COVID Admissions
  - Staffed Adult ICU Bed Occupancy
  - Day of the Week  
- **Target Variable**: Inpatient Beds Used for COVID Patients

---

## 📦 Tech Stack

- Python 🐍
- Pandas
- Scikit-learn
- Matplotlib

---

## 💻 Getting Started

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Simran0192004/hospital-bed-occupancy-predictor.git
cd hospital-bed-occupancy-predictor
pip install -r requirements.txt
