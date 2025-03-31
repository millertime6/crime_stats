import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from category_encoders import TargetEncoder

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('BPD_arrests.csv')
    return data

data = load_data()

# Drop rows with missing target values or essential features
data = data.dropna(subset=['IncidentOffence', 'ArrestDateTime', 'Age', 'Gender', 'District'])

# Strip whitespace from relevant columns
data['IncidentOffence'] = data['IncidentOffence'].str.strip()
data['District'] = data['District'].str.strip()
data['Neighborhood'] = data['Neighborhood'].str.strip()

# Store original categorical values for UI before encoding
district_options = sorted(data['District'].unique())
neighborhood_options = sorted(data['Neighborhood'].fillna('Unknown').unique())

# Fill or drop other missing values
data['Neighborhood'] = data['Neighborhood'].fillna('Unknown')
data['ChargeDescription'] = data['ChargeDescription'].fillna('Unknown')

# Convert ArrestDateTime to datetime
data['ArrestDateTime'] = pd.to_datetime(data['ArrestDateTime'], errors='coerce')

# Create new time-based features
data['Hour'] = data['ArrestDateTime'].dt.hour
data['DayOfWeek'] = data['ArrestDateTime'].dt.dayofweek
data['Month'] = data['ArrestDateTime'].dt.month
data['TimeOfDay'] = pd.cut(data['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)

# Encode categorical variables
cat_features = ['Gender', 'District', 'Neighborhood', 'TimeOfDay']
encoder = TargetEncoder()
# Encode the target variable first
label_encoder = LabelEncoder()
data['IncidentOffence_encoded'] = label_encoder.fit_transform(data['IncidentOffence'])

# Use the encoded target variable for TargetEncoder
data[cat_features] = encoder.fit_transform(data[cat_features], data['IncidentOffence_encoded'])

# Split the data
# Define features and target
features = ['Age', 'Gender', 'District', 'Neighborhood', 'Hour', 'DayOfWeek', 'Month', 'TimeOfDay']
target = 'IncidentOffence'

# Filter only necessary columns and drop any remaining NaNs
data_model = data[features + [target]].dropna()

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_model[features], data_model[target], test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation result
st.write(f"Model Accuracy: {accuracy:.2f}")

# Prediction form
st.header("Predict Crime Type")
with st.form("crime_prediction_form"):
    age_input = st.number_input("Offender Age", min_value=10, max_value=100, value=30)

    gender_display = st.selectbox("Gender", ['Male', 'Female'])
    gender_input = 'M' if gender_display == 'Male' else 'F'

    district_input = st.selectbox("District", district_options)
    neighborhood_input = st.selectbox("Neighborhood", neighborhood_options)

    hour_input = st.number_input("Hour of Arrest (1–24)", min_value=1, max_value=24, value=12)
    day_of_week_input = st.number_input("Day of Week (1=Mon)", min_value=1, max_value=7, value=1)
    month_input = st.number_input("Month", min_value=1, max_value=12, value=1)

    # Adjust TimeOfDay bin logic if needed
    time_of_day_input = pd.cut([hour_input], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)[0]

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            'Age': [age_input],
            'Gender': [gender_input],
            'District': [district_input],
            'Neighborhood': [neighborhood_input],
            'Hour': [hour_input],
            'DayOfWeek': [day_of_week_input - 1],
            'Month': [month_input],
            'TimeOfDay': [time_of_day_input]
        })

        input_data[cat_features] = encoder.transform(input_data[cat_features])
        prediction_label = model.predict(input_data)[0]
        st.success(f"Predicted Crime Type: {prediction_label}")
