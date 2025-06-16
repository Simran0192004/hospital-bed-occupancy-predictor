#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing the dataset
df = pd.read_csv('hospital_data.csv')

#Dropping Unnecessary columns
df = df.drop(columns = ['hospital_pk', 'hospital_name', 'fips', 'address', 'city', 'zip',
    'geocoded_hospital_address'], errors='ignore')

#Converting date and creating new features
df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
df['day_of_week'] = df['date'].dt.dayofweek

columns_needed = [
    'previous_day_admission_adult_covid_confirmed',
    'staffed_adult_icu_bed_occupancy',
    'total_adult_patients_hospitalized_confirmed_covid',
    'inpatient_beds_used',
    'inpatient_beds_used_covid'
]
df = df.dropna(subset=columns_needed)

#Splitting the dataset into features and target
X = df[[ 'previous_day_admission_adult_covid_confirmed',  
    'staffed_adult_icu_bed_occupancy',  
    'total_adult_patients_hospitalized_confirmed_covid',  
    'inpatient_beds_used',  
    'day_of_week' ]]
y = df['inpatient_beds_used_covid']

#Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Make Predictions
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.values.reshape(len(y_test),1)), axis=1))

#Plotting actual vs Predicted Value
plt.scatter(y_test, y_pred, color='skyblue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Hospital Bed Occupancy')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()